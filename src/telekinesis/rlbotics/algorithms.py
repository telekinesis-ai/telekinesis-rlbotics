"""Reinforcement learning algorithms.

Attributes:
    ALGORITHMS: Algorithm classes that can be named by an algorithm config.
    OPTIMIZERS: Optimizer classes that can be named by an algorithm config.
"""

from __future__ import annotations

from itertools import chain

import torch
import torch.nn as nn
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.config import PPOConfig
from telekinesis.rlbotics.models import CNNModel, MLPModel
from telekinesis.rlbotics.rollout import RolloutBuffer, Transition
from telekinesis.rlbotics.symmetry import Symmetry

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel | CNNModel
    """The actor model."""

    critic: MLPModel | CNNModel
    """The critic model."""

    def __init__(
        self,
        cfg: PPOConfig,
        actor: MLPModel | CNNModel,
        critic: MLPModel | CNNModel,
        storage: RolloutBuffer,
        obs_groups: dict[str, list[str]] | None = None,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
        env=None,
    ) -> None:
        """Initialize the algorithm from its config, with the models and storage it optimizes.

        Args:
            cfg: Algorithm configuration holding every hyperparameter.
            actor: The actor model, already sized for the observation set it consumes.
            critic: The critic model, already sized for the observation set it consumes.
            storage: Rollout storage for the collected experience.
            obs_groups: Mapping from observation set to the observation groups it consumes. Defaults to None, in which
                case both models receive the single observation group returned by the environment.
            device: Device for computation.
            multi_gpu_cfg: Distributed training settings, or None for single-device training.
            env: The environment being trained on. Only the symmetry extension needs it, to hand to
                the mirror function. Defaults to None.

        Raises:
            ValueError: If the config requests an unimplemented extension, or symmetry with a recurrent policy.
        """
        self.cfg = cfg

        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # Unimplemented extensions
        if cfg.rnd_cfg is not None:
            raise ValueError("The RND extension is configured but not implemented. Set 'rnd_cfg' to None.")

        # Symmetry extension
        if cfg.symmetry_cfg is not None and (
            getattr(actor, "is_recurrent", False) or getattr(critic, "is_recurrent", False)
        ):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(cfg.symmetry_cfg, env=env) if cfg.symmetry_cfg else None

        # Observation sets consumed by the actor and the critic. If unset, both models receive the single observation
        # group returned by the environment.
        self.obs_groups = obs_groups

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic = self.critic

        # Create the optimizer
        self.optimizer = OPTIMIZERS[cfg.optimizer.lower()](
            chain(self.actor.parameters(), self.critic.parameters()), lr=cfg.learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = Transition()

        # The learning rate is the only hyperparameter that changes during training, under the adaptive schedule. Every
        # other one is read from the config where it is used.
        self.learning_rate = cfg.learning_rate

        # Diagnostics from the most recent update, published for logging
        self.diagnostics: dict[str, float] = {}
        self._explained_variance = 0.0

    def get_obs_set(self, obs: TensorDict | torch.Tensor, set_name: str) -> torch.Tensor:
        """Extract the observation tensor consumed by one model.

        The groups configured for the observation set are concatenated along the feature dimension. A set naming a
        single group is returned unchanged, which is what image observations require.

        Args:
            obs: Observations from the environment, or an already extracted tensor.
            set_name: Observation set to extract, either ``"actor"`` or ``"critic"``.

        Returns:
            The observation tensor for the requested set.
        """
        if not isinstance(obs, TensorDict):
            return obs
        # Without an explicit mapping, fall back to the single observation group from the environment
        if self.obs_groups is None:
            return obs["observation"] if "observation" in obs else next(iter(obs.values()))
        groups = self.obs_groups[set_name]
        if len(groups) == 1:
            return obs[groups[0]]
        return torch.cat([obs[group] for group in groups], dim=-1)

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Extract the observation set of each model
        actor_obs = self.get_obs_set(obs, "actor")
        critic_obs = self.get_obs_set(obs, "critic")

        # Compute the actions and values
        self.transition.actions = self.actor(actor_obs, stochastic=True).detach()
        self.transition.values = self.critic(critic_obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        if hasattr(self.actor, 'output_distribution_params'):
            self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        else:
            self.transition.distribution_params = ()
        # Record observations before env.step()
        self.transition.observations = actor_obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers with the observation set each model consumes
        self.actor.update_normalization(self.get_obs_set(obs, "actor"))
        self.critic.update_normalization(self.get_obs_set(obs, "critic"))

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Bootstrapping on time outs. An episode cut off by the time limit is not really over, so the
        # value of the state it was cut off in is added back to the reward.
        #
        # This follows the VecEnv contract: rewards and dones are (num_envs,) while values are
        # (num_envs, 1), so the product is squeezed back to (num_envs,) before the in-place add.
        if "time_outs" in extras:
            self.transition.rewards += self.cfg.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        if hasattr(self.actor, 'reset'):
            self.actor.reset(dones)
        if hasattr(self.critic, 'reset'):
            self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute values for the last step from the critic's observation set
        last_values = self.critic(self.get_obs_set(obs, "critic")).detach()
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.cfg.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.cfg.gamma * self.cfg.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # How much of the return variance the critic explained on this rollout. 1.0 is perfect,
        # 0.0 is no better than predicting the mean, and negative means worse than that.
        with torch.inference_mode():
            return_var = st.returns.var()
            self._explained_variance = float(
                1.0 - (st.returns - st.values).var() / return_var if return_var > 0 else 0.0
            )

        # Normalize the advantages if per minibatch normalization is not used
        if not self.cfg.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_kl = 0.0
        mean_clip_fraction = 0.0

        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # Get mini-batch generator
        generator = self.storage.mini_batch_generator(self.cfg.num_mini_batches, self.cfg.num_learning_epochs)

        # Iterate over mini-batches
        for batch in generator:
            # Get batch size from observations
            if isinstance(batch.observations, TensorDict):
                original_batch_size = batch.observations.batch_size[0]
            else:
                original_batch_size = batch.observations.shape[0]

            # Check if we should normalize advantages per mini-batch
            if self.cfg.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation if enabled
            if self.symmetry:
                self.symmetry.augment_batch(batch, original_batch_size)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with new parameters
            # The batch stores each model's observation set, so no extraction is needed here
            actor_obs = batch.observations
            critic_obs = batch.critic_observations if batch.critic_observations is not None else actor_obs

            self.actor(actor_obs, stochastic=True)
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(critic_obs)
            # Note: We only keep the following tensors for the original samples in case of symmetry augmentation
            if hasattr(self.actor, 'output_distribution_params'):
                distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            else:
                distribution_params = ()
            if hasattr(self.actor, 'output_entropy'):
                entropy = self.actor.output_entropy[:original_batch_size]
            else:
                entropy = torch.zeros(original_batch_size, device=self.device)

            # Measure how far this update moved the policy. This is the diagnostic that catches a
            # blown-up update, so it is computed on every schedule, not only the adaptive one.
            has_distribution = (
                getattr(self.actor, "distribution", None) is not None
                and batch.old_distribution_params is not None
            )
            kl_mean = None
            if has_distribution:
                with torch.inference_mode():
                    kl = self.actor.distribution.kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size
                mean_kl += kl_mean.item()

            # Adapt the learning rate to keep the policy change near the target
            if kl_mean is not None and self.cfg.desired_kl is not None and self.cfg.schedule == "adaptive":
                with torch.inference_mode():
                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.cfg.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.cfg.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Fraction of samples whose ratio left the trust region. A spike here means the update
            # is being throttled by the clip, which is where runaway policy changes show up.
            with torch.inference_mode():
                mean_clip_fraction += ((ratio - 1.0).abs() > self.cfg.clip_param).float().mean().item()

            # Value function loss
            if self.cfg.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.cfg.value_loss_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

            # Symmetry loss
            if self.symmetry:
                symmetry_loss = self.symmetry.compute_loss(self.actor, batch, original_batch_size)
                if self.symmetry.cfg.use_mirror_loss:
                    loss = loss + self.symmetry.cfg.mirror_loss_coeff * symmetry_loss

            # Gradient clipping only bounds the norm of finite gradients; a NaN or Inf loss (from a
            # reward or observation spike) sails through it unclamped and corrupts every parameter
            # on the next step. Skip the update instead of training on it.
            if not torch.isfinite(loss):
                logger.warning(f"non-finite loss ({loss.item()}), skipping this mini-batch update")
                continue

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.cfg.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.cfg.max_grad_norm)
            self.optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates

        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }

        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        # Diagnostics for spotting an unstable update, kept apart from the losses
        self.diagnostics = {
            "kl": mean_kl / num_updates,
            "clip_fraction": mean_clip_fraction / num_updates,
            "explained_variance": self._explained_variance,
        }

        # Clear the storage
        self.storage.clear()

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_state_dict": self._raw_critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def get_learning_rate(self) -> float:
        """Get the current learning rate."""
        return self.learning_rate

    def get_action_std(self) -> float | None:
        """Get the mean action standard deviation, or ``None`` if the actor is deterministic or has not acted yet."""
        distribution = getattr(self._raw_actor, "distribution", None)
        if distribution is None:
            return None
        try:
            return distribution.std.mean().item()
        except AttributeError:
            # The distribution has not been updated with a model output yet
            return None

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rlbotics.algorithms.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic = compile_model(self._raw_critic, mode)  # type: ignore

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic.state_dict()]

        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic.load_state_dict(model_params[1])


    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel


def check_nan(obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor) -> None:
    """Raise ``ValueError`` if any environment output contains NaN."""
    for key, tensor in obs.items():
        if torch.isnan(tensor).any():
            raise ValueError(
                f"The observation group '{key}' returned by the environment contains NaN values. This usually indicates"
                " a bug in the environment's step() or reset() function."
            )
    if torch.isnan(rewards).any():
        raise ValueError(
            "The rewards returned by the environment contain NaN values. This usually indicates a bug in the"
            " environment's reward computation."
        )
    if torch.isnan(dones).any():
        raise ValueError(
            "The dones returned by the environment contain NaN values. This usually indicates a bug in the"
            " environment's termination logic."
        )


def compile_model(model: torch.nn.Module, mode: str | None = None) -> torch.nn.Module:
    """Wrap a model with :func:`torch.compile`, validating the compile mode.

    Args:
        model: The model to compile.
        mode: The :func:`torch.compile` mode. CUDA-graph modes (``"reduce-overhead"``, ``"max-autotune"``) are rejected
        because they are incompatible with the multi-model forward patterns used by the algorithms (graph replay
        overwrites the previous call's output buffer). Use ``"default"`` or ``"max-autotune-no-cudagraphs"`` instead.
        Defaults to ``None``, in which case compilation is disabled.

    Returns:
        The compiled model, or the original model if ``mode`` is ``None``.

    Raises:
        ValueError: If ``mode`` is one of the unsupported CUDA-graph modes.
    """
    if mode is None:
        return model
    if mode in ("reduce-overhead", "max-autotune"):
        raise ValueError(
            f"torch_compile_mode='{mode}' uses CUDA graphs which are incompatible with the algorithms' multi-model "
            f"forward pattern. Use 'default' or 'max-autotune-no-cudagraphs', or set to None to disable."
        )
    return torch.compile(model, mode=mode)  # type: ignore


# Algorithm classes that can be named by an algorithm config
ALGORITHMS = {
    "PPO": PPO,
}
