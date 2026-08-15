"""On-policy runner for reinforcement learning algorithms."""

from __future__ import annotations

import os
import time
from pathlib import Path

import torch
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.algorithms import ALGORITHMS, check_nan
from telekinesis.rlbotics.checkpoint import CheckpointManager
from telekinesis.rlbotics.config import (
    CNNConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
)
from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.logger import Logger, VideoLogger
from telekinesis.rlbotics.models import CNNModel, MLPModel
from telekinesis.rlbotics.rollout import RolloutBuffer
from telekinesis.rlbotics.utils import resolve_callable, resolve_device, set_seed


class _ScaledPolicy(torch.nn.Module):
    """Wraps a policy so the exported graph emits actions in the environment's range.

    The policy itself works in ``[-1, 1]``; the environment adapter normally does the clipping and
    the affine scaling onto its own bounds. Folding both into the export means the file does not
    have to be paired with a copy of those bounds to be used correctly.
    """

    def __init__(self, policy: torch.nn.Module, low: torch.Tensor, high: torch.Tensor) -> None:
        """Store the policy and the action range.

        Args:
            policy: The export-friendly deterministic policy.
            low: Lower action bound.
            high: Upper action bound.
        """
        super().__init__()
        self.policy = policy
        self.register_buffer("low", low)
        self.register_buffer("high", high)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the action for an observation, scaled to the environment's range."""
        action = self.policy(observation).clamp(-1.0, 1.0)
        return self.low + 0.5 * (action + 1.0) * (self.high - self.low)


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms.

    Manages the training loop, environment interactions, and model updates for on-policy
    reinforcement learning algorithms such as PPO.
    """

    def __init__(
        self,
        env: VecEnv,
        runner_cfg: OnPolicyRunnerConfig,
        device: str = "cpu",
    ) -> None:
        """Construct the runner, its models, and its algorithm from the config.

        Logging and checkpointing come from ``runner_cfg.logger`` and ``runner_cfg.checkpoint``. When
        the checkpoint config asks to resume, that happens here, so the runner is ready to continue
        training as soon as it is built.

        If ``runner_cfg.seed`` is set, it is applied first, before the actor and critic are built, so
        model initialization is reproducible too. This constructs the ``OnPolicyRunner`` class
        directly; to build whichever runner class ``runner_cfg.class_name`` names, use
        :func:`create_runner`.

        Args:
            env: Vectorized environment for parallel experience collection.
            runner_cfg: Runner configuration, including the logger, checkpoint, algorithm, actor and
                critic configs.
            device: Device for computation ("cpu" or "cuda:0", etc.).
        """
        if runner_cfg.seed is not None:
            set_seed(runner_cfg.seed)

        self.env = env
        self.runner_cfg = runner_cfg
        self.device = resolve_device(device)

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Get observations for algorithm construction and check that every configured group exists
        obs = self.env.get_observations()
        self.obs_groups = runner_cfg.obs_groups
        for set_name, groups in self.obs_groups.items():
            for group in groups:
                if group not in obs:
                    raise ValueError(
                        f"Observation group '{group}' of set '{set_name}' was not found in the observations from the"
                        f" environment. Available observations: {list(obs.keys())}."
                    )

        # Create actor and critic
        actor = self._build_model(runner_cfg.actor, obs, "actor", self.env.num_actions)
        critic = self._build_model(runner_cfg.critic, obs, "critic", 1)

        # Share CNN encoders between actor and critic if requested
        if runner_cfg.algorithm.share_cnn_encoders:
            if not (isinstance(actor, CNNModel) and isinstance(critic, CNNModel)):
                raise ValueError("'share_cnn_encoders' requires both the actor and the critic to be CNN models.")
            critic.cnn = actor.cnn

        # Create the rollout storage, giving the critic its own buffer when it observes a different set
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=self.env.num_envs,
            num_transitions_per_env=runner_cfg.num_steps_per_env,
            obs_shape=self._obs_set_shape(obs, "actor"),
            actions_shape=(self.env.num_actions,),
            device=self.device,
            critic_obs_shape=self._obs_set_shape(obs, "critic"),
        )

        # Create algorithm
        alg_cfg = runner_cfg.algorithm
        if isinstance(alg_cfg, PPOConfig):
            # A class passed directly is used as-is, a name is looked up
            alg_class = alg_cfg.class_name if callable(alg_cfg.class_name) else ALGORITHMS[alg_cfg.class_name]
        else:
            raise ValueError(f"Unsupported algorithm config type: {type(alg_cfg).__name__}.")

        self.alg = alg_class(
            cfg=alg_cfg,
            actor=actor,
            critic=critic,
            storage=storage,
            obs_groups=self.obs_groups,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            env=self.env,
        )

        # Compile the models if requested
        self.alg.compile(runner_cfg.torch_compile_mode)

        # Create the logger, which decides the run directory: <log_dir>/<experiment>/<timestamp>
        self.logger = Logger(
            cfg=runner_cfg.logger,
            num_envs=self.env.num_envs,
            num_steps_per_env=runner_cfg.num_steps_per_env,
            device=self.device,
        )
        self.log_dir = self.logger.log_dir

        # Record what produced this run, for reproducing it later
        self.logger.save_config(runner_cfg.to_dict())

        # Checkpoints go flat into the run directory, beside the event file and the config dump
        self.checkpoints = CheckpointManager(runner_cfg.logger, run_dir=self.log_dir)

        # A video of every checkpoint, rendered from the training rollout itself
        self.video: VideoLogger | None = None
        if runner_cfg.logger.log_video:
            recorder = VideoLogger(self.env, obs_groups=self.obs_groups["actor"])
            if recorder.can_render:
                self.video = recorder
            else:
                logger.warning(
                    f"   'log_video' is on but {type(self.env).__name__} renders nothing, so no "
                    'video will be written. Build the environment with render_mode="rgb_array".'
                )

        self.current_learning_iteration = 0

        # Continue from a checkpoint if one was requested
        resume_from = self.checkpoints.resume_path()
        if resume_from is not None:
            self.load(str(resume_from))
            self.resumed_from: Path | None = resume_from
        else:
            self.resumed_from = None

    def _obs_set_shape(self, obs: TensorDict, set_name: str) -> tuple[int, ...]:
        """Return the per-environment shape of an observation set.

        A set naming a single group keeps that group's shape, which is what image observations require. Several groups
        are concatenated along the feature dimension.

        Args:
            obs: Observations from the environment.
            set_name: Observation set to measure, either ``"actor"`` or ``"critic"``.

        Returns:
            The observation shape, excluding the environment dimension.

        Raises:
            ValueError: If a group holds separate terms rather than one tensor, or if a set
                combines several groups that are not flat vectors.
        """
        groups = self.obs_groups[set_name]
        if len(groups) == 1:
            value = obs[groups[0]]
            # A group whose terms the simulator publishes separately rather than concatenated
            # arrives nested, and a nested value has only the environment dimension. Caught here
            # because the shape that survives is (), which reads downstream as a malformed tensor
            # and draws an error about image observations that has nothing to do with it.
            if not isinstance(value, torch.Tensor):
                terms = sorted(str(key) for key in value.keys())
                raise ValueError(
                    f"Observation group '{groups[0]}' holds separate terms {terms} rather than one"
                    f" tensor, so observation set '{set_name}' has no shape to size a model from."
                    " The task publishes this group unconcatenated; concatenate its terms in the"
                    " task's observation configuration to train on it."
                )
            return tuple(value.shape[1:])

        for group in groups:
            if obs[group].dim() != 2:
                raise ValueError(
                    f"Observation set '{set_name}' concatenates several groups, which requires every group to be a"
                    f" flat vector, but group '{group}' has shape {tuple(obs[group].shape)}. Give this set a single"
                    f" group, or flatten the observation in the environment."
                )
        return (sum(int(obs[group].shape[-1]) for group in groups),)

    def _build_model(
        self,
        cfg: MLPConfig | CNNConfig,
        obs: TensorDict,
        set_name: str,
        output_dim: int,
    ) -> MLPModel | CNNModel:
        """Build the actor or the critic from its config.

        The model is sized from the observation groups assigned to ``set_name``, so it always matches the observations
        it will receive.

        Args:
            cfg: Model configuration.
            obs: Observations from the environment.
            set_name: Observation set the model consumes, either ``"actor"`` or ``"critic"``.
            output_dim: Dimension of the model output.

        Returns:
            The model, moved to the runner's device.

        Raises:
            ValueError: If the config type is unsupported, or a CNN model is configured for an observation set that is
                not a single group of images.
        """
        obs_shape = self._obs_set_shape(obs, set_name)

        if isinstance(cfg, CNNConfig):
            groups = self.obs_groups[set_name]
            if len(groups) != 1 or len(obs_shape) != 3:
                raise ValueError(
                    f"A CNN model requires observation set '{set_name}' to name exactly one group of images with"
                    f" shape (channels, height, width), got groups {groups} with shape {obs_shape}."
                )
            model = CNNModel(
                cfg=cfg,
                input_dim=(obs_shape[1], obs_shape[2]),
                input_channels=obs_shape[0],
                output_dim=output_dim,
                obs_group=groups[0],
            )
        elif isinstance(cfg, MLPConfig):
            if len(obs_shape) != 1:
                raise ValueError(
                    f"An MLP model requires observation set '{set_name}' to be a flat vector, got shape {obs_shape}."
                    f" Configure a CNN model for image observations."
                )
            model = MLPModel(cfg=cfg, input_dim=obs_shape[0], output_dim=output_dim)
        else:
            raise ValueError(f"Unsupported model config type for the {set_name}: {type(cfg).__name__}.")

        return model.to(self.device)

    def learn(
        self, num_learning_iterations: int | None = None, init_at_random_ep_len: bool = False
    ) -> None:
        """Run the learning loop.

        With ``logger.log_video`` on, the rollout of every iteration that ends in a checkpoint is
        rendered and written beside it as ``model_<iteration>.mp4``, so there is a clip of what the
        policy was doing at the checkpoint it belongs to.

        Args:
            num_learning_iterations: Number of learning iterations to execute. Defaults to None,
                which uses the config's ``num_learning_iterations``, so a run configured entirely
                from a file needs no argument here.
            init_at_random_ep_len: Whether to randomize initial episode lengths for exploration.
        """
        if num_learning_iterations is None:
            num_learning_iterations = self.runner_cfg.num_learning_iterations
        # Randomize initial episode lengths (for exploration). The counter is optional on the VecEnv
        # interface, since it belongs to whatever owns the episode, so an environment without one
        # gets told rather than silently starting every episode at zero.
        if init_at_random_ep_len:
            episode_lengths = getattr(self.env, "episode_length_buf", None)
            if episode_lengths is None:
                logger.warning(
                    "   init_at_random_ep_len was requested, but this environment does not expose "
                    "'episode_length_buf', so every episode starts at step 0."
                )
            else:
                max_ep_len = getattr(self.env, "max_episode_length", 1000)
                self.env.episode_length_buf = torch.randint_like(
                    episode_lengths, high=int(max_ep_len)
                )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()

        # Every rank has to start from the same weights. The gradients are averaged across ranks, so
        # ranks that began from different random initializations would each be descending on a
        # different function and the average would mean nothing.
        if self.is_distributed:
            logger.info(f"   Synchronizing parameters for rank {self.gpu_global_rank}.")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        if self.logger.writer is not None:
            self.logger.writer.flush()

        # Start training - handle resuming from previous checkpoints
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations

        frames: list | None = None
        for it in range(start_it, total_it):
            start = time.time()

            # Record only the rollouts that end in a checkpoint, so the cost is one render per step
            # on one iteration in save_interval rather than on every one
            capture = self.video is not None and (
                self.checkpoints.should_save(it) or it == total_it - 1
            )
            frames = [] if capture else frames

            # Rollout - collect experience from environment
            with torch.inference_mode():
                for _ in range(self.runner_cfg.num_steps_per_env):
                    # Sample actions from policy
                    actions = self.alg.act(obs)
                    # Step the environment, on whichever device it simulates on. That is usually this
                    # runner's device, but an environment is free to differ — a GPU policy driving a
                    # CPU simulator, say — and the move belongs here rather than in every adapter.
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    # Catch environment bugs early, before they propagate into the models
                    if self.runner_cfg.check_for_nan:
                        check_nan(obs, rewards, dones)

                    # Process environment step in algorithm
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Log episode information
                    if "log" in extras:
                        self.logger.process_env_step(rewards, dones, extras.get("log"))
                    else:
                        self.logger.process_env_step(rewards, dones)

                    if capture:
                        frames.append(self.video.frame())

                collect_time = time.time() - start
                start = time.time()

                # Compute returns (value function bootstrapping)
                self.alg.compute_returns(obs)

            # Update policy
            loss_dict = self.alg.update()

            learn_time = time.time() - start

            # Advanced before anything is written, because _state_dict() records it. A checkpoint
            # named model_<it>.pt that carried iteration <it - 1> inside would make --resume replay an
            # iteration, and would misreport which iteration the best policy came from.
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                iteration=it,
                total_iterations=total_it,
                start_iteration=start_it,
                collect_time=collect_time,
                learn_time=learn_time,
                losses=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_action_std(),
                diagnostics=getattr(self.alg, "diagnostics", None),
                # The runner does the reporting, so it decides whether to print at all
                print_interval=(
                    self.runner_cfg.logger.log_interval if self.runner_cfg.verbose else 0
                ),
            )

            # Keep the best policy, scored on the mean reward of the recent episodes. Checked every
            # iteration rather than on the save schedule, so a peak between two scheduled saves is
            # not lost when the reward later drops
            mean_reward = self.logger.mean_reward
            if mean_reward is not None:
                self.checkpoints.save_best(self._state_dict(), mean_reward, it)

            # Save model checkpoint
            if self.checkpoints.should_save(it):
                self._write_checkpoint(it, frames)

        # Save the final model after training
        if self.checkpoints.enabled:
            self._write_checkpoint(self.current_learning_iteration, frames)
        if self.logger.writer is not None:
            self.logger.writer.flush()

    def _write_checkpoint(self, iteration: int, frames: list | None) -> Path | None:
        """Write a checkpoint and, when there are frames, the video of the rollout beside it.

        Args:
            iteration: Iteration the checkpoint belongs to.
            frames: Rendered frames of that iteration's rollout, or None when not recording.

        Returns:
            The checkpoint written, or None if checkpointing is disabled.
        """
        save_path = self.checkpoints.save(self._state_dict(), iteration)
        if save_path is not None and self.video is not None and frames:
            video_path = self.video.write(frames, save_path.with_suffix(".mp4"))
            if self.runner_cfg.verbose:
                logger.info(f"   Wrote {video_path}")
        return save_path

    def _state_dict(self) -> dict:
        """Collect everything needed to resume training later.

        Returns:
            The algorithm's state plus the current iteration.
        """
        state = self.alg.save()
        state["iteration"] = self.current_learning_iteration
        # Carried so a later run measures "best" against the whole experiment, not just itself
        state["best_metric"] = self.checkpoints.best_metric
        return state

    def save(self, path: str) -> None:
        """Save the models and training state to a specific path.

        Checkpoints written during training go through :class:`CheckpointManager` instead; this is
        for saving on demand.

        Args:
            path: Path where the model should be saved.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self._state_dict(), path)

    def load(self, path: str, strict: bool = True) -> None:
        """Load the models and training state from disk.

        Args:
            path: Path to load the model from.
            strict: Whether to strictly enforce matching state dict keys.
        """
        loaded_dict = torch.load(path, weights_only=False)

        if hasattr(self.alg, "load"):
            self.alg.load(loaded_dict, load_cfg=None, strict=strict)

        if "iteration" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iteration"]

        # Pick the best score back up, so continuing does not overwrite a better checkpoint with a
        # worse one just because this run started from scratch on the metric
        if loaded_dict.get("best_metric") is not None:
            self.checkpoints.best_metric = loaded_dict["best_metric"]

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy for inference.

        Args:
            device: Device to move the policy to. If None, uses the runner's device.

        Returns:
            The policy model in evaluation mode.
        """
        if device is None:
            device = self.device

        self.alg.eval_mode()

        policy = self.alg.get_policy()
        return policy.to(device)

    def get_policy_snapshot(self, device: str | None = None) -> MLPModel | CNNModel:
        """Return an independent copy of the current policy, safe to use during training.

        :meth:`get_inference_policy` hands back the actual actor being trained: switching it to
        eval mode and moving it to another device reaches into the training loop. ``copy.deepcopy``
        cannot stand in for that either: after a training step, the actor's distribution caches a
        ``torch.distributions.Normal`` built from its latest forward pass, whose tensors are
        non-leaf and still require grad, which the tensor deepcopy protocol rejects. So this rebuilds
        a fresh actor from the runner's config instead and copies over only the trained weights,
        which sidesteps that cache entirely. Calling this from an ``on_checkpoint`` callback — to
        record a video of the policy at its current iteration, say — leaves training untouched.

        Args:
            device: Device to move the copy to. Defaults to None, meaning the runner's device.

        Returns:
            A copy of the policy, in evaluation mode.
        """
        if device is None:
            device = self.device

        obs = self.env.get_observations()
        snapshot = self._build_model(self.runner_cfg.actor, obs, "actor", self.env.num_actions)
        snapshot.load_state_dict(self.alg.get_policy().state_dict())
        snapshot = snapshot.to(device)
        snapshot.eval()
        return snapshot

    def _policy_for_export(self, from_best: bool) -> MLPModel | CNNModel:
        """Return the policy to export, preferring the run's best checkpoint.

        The weights are loaded into a copy rather than into the actor being trained, so exporting
        never changes what a subsequent call to :meth:`learn` continues from.

        Args:
            from_best: Whether to look for a best checkpoint at all.

        Returns:
            The policy to export, in evaluation mode on the CPU.
        """
        policy = self.get_policy_snapshot(device="cpu")

        best = self.checkpoints.best() if from_best else None
        if best is None:
            if from_best and self.runner_cfg.verbose:
                logger.info("   No best checkpoint to export, exporting the current policy.")
            return policy

        state = torch.load(best, weights_only=False, map_location="cpu")
        policy.load_state_dict(state["actor_state_dict"])
        policy.eval()
        if self.runner_cfg.verbose:
            metric, iteration = state.get("best_metric"), state.get("best_iteration")
            logger.info(
                f"   Exporting {best.name}, which scored {metric:.2f} at iteration {iteration}."
                if metric is not None
                else f"   Exporting {best.name}."
            )
        return policy

    # Two, not one. This tensor is only a tracing example, but a size-one dimension cannot be made
    # dynamic: torch specializes it to the literal 1 and the ONNX graph then accepts a batch of one
    # and nothing else, which is exactly what the dynamic batch axis in export_policy_to_onnx is
    # there to prevent. It fails silently -- no error at export, only an InvalidArgument from
    # onnxruntime at deployment -- and it is version-dependent, so it does not show up on every
    # torch the package supports.
    EXPORT_BATCH = 2

    def _export_dummy_input(self, policy: MLPModel | CNNModel) -> torch.Tensor:
        """Build a dummy input matching what the policy expects.

        Args:
            policy: The policy to export.

        Returns:
            A dummy input tensor whose batch dimension is :data:`EXPORT_BATCH`, which is large
            enough for that dimension to survive as a dynamic one.
        """
        if isinstance(policy, CNNModel):
            return torch.randn(self.EXPORT_BATCH, policy.input_channels, *policy.input_dim)
        return torch.randn(self.EXPORT_BATCH, policy.input_dim)

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the policy to TorchScript format.

        The exported module takes an observation and returns the deterministic action, so it carries no sampling and no
        distribution parameters.

        Args:
            path: Directory to save the exported model.
            filename: Filename for the exported model.
        """
        policy = self.alg.get_policy().to("cpu")
        policy.eval()

        os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace the export-friendly deterministic wrapper, which starts in training mode, and save it
        traced_model = torch.jit.trace(policy.as_jit().eval(), self._export_dummy_input(policy))
        traced_model.save(save_path)

    def export(
        self, path: str | None = None, filename: str = "policy.onnx", from_best: bool = True
    ) -> Path:
        """Export the best policy for deployment and return the file written.

        The graph is self-contained: it takes a raw observation, applies the observation
        normalization the policy was trained with, produces the deterministic action, and scales it
        onto the environment's action range. So a consumer needs nothing but the file, which is what
        :class:`~telekinesis.rlbotics.policy.Policy` loads.

        What gets exported is the best checkpoint of the run, not the policy left in memory when
        training stopped. Those differ whenever the reward peaked and then fell back, which for a
        long locomotion run is the common case, and shipping the final policy in that situation means
        deploying a worse one than was trained.

        Args:
            path: Directory to write to. Defaults to None, meaning this run's directory.
            filename: Name of the exported file.
            from_best: Whether to export the run's best checkpoint. Defaults to True. False exports
                the current policy, which is what you want when comparing where training ended up.

        Returns:
            The path of the exported policy.

        Raises:
            ValueError: If there is no directory to write to, which happens when logging is off and
                no path is given.
        """
        directory = Path(path) if path is not None else (
            Path(self.log_dir) if self.log_dir is not None else None
        )
        if directory is None:
            raise ValueError(
                "There is nowhere to export to: this run has no log directory. Pass a path, or set "
                "'logger.log_dir'."
            )

        policy = self._policy_for_export(from_best)

        directory.mkdir(parents=True, exist_ok=True)
        save_path = directory / filename

        # An environment that bounds its actions gets the clipping and scaling baked in, so the
        # exported policy emits actions the environment accepts directly
        exported = policy.as_onnx()
        low = getattr(self.env, "action_low", None)
        high = getattr(self.env, "action_high", None)
        if low is not None and high is not None:
            exported = _ScaledPolicy(exported, low.to("cpu"), high.to("cpu"))
        # The wrappers are freshly constructed, so they start in training mode
        exported.eval()

        torch.onnx.export(
            exported,
            (self._export_dummy_input(policy),),
            str(save_path),
            input_names=["observation"],
            output_names=["action"],
            opset_version=18,
            dynamo=True,
            # Keep the weights in the file. Splitting them into a '<filename>.data' sidecar is the
            # default, and then the graph cannot be moved or loaded on its own
            external_data=False,
            # A deployment may act on one observation or on a batch of them. Given positionally,
            # since the wrapper's argument name is not part of this method's contract
            dynamic_shapes=({0: torch.export.Dim("batch")},),
        )
        return save_path

    def export_policy_to_onnx(
        self, path: str, filename: str = "policy.onnx", verbose: bool = False
    ) -> None:
        """Export the policy to ONNX format.

        Deprecated in favour of :meth:`export`, which returns the path it wrote and bakes the action
        scaling into the graph.

        Args:
            path: Directory to save the exported model.
            filename: Filename for the exported model.
            verbose: Whether to print verbose output. Unused.
        """
        self.export(path=path, filename=filename)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-GPU training.

        The distributed settings are derived from the launch environment rather than the config, and are stored in
        :attr:`multi_gpu_cfg` for the algorithm to consume. When the launcher reports more than one process this also
        brings up the NCCL process group and binds this process to its own GPU. Both are prerequisites, not
        conveniences: the algorithm's gradient averaging and :meth:`learn`'s parameter broadcast are collective calls
        that fail on an uninitialized group.

        The device is retargeted rather than rejected. :func:`~telekinesis.rlbotics.utils.resolve_device` maps a bare
        "cuda" onto "cuda:0", so a launch that hands every process the same device string would otherwise train every
        rank on GPU 0. Note that this only moves the models and the rollout storage: the environment is built before the
        runner and stays wherever its caller put it, so a training script under ``torchrun`` should still pass
        ``cuda:$LOCAL_RANK`` to the environment to keep the simulation off one GPU.

        Raises:
            ValueError: If the ranks reported by the launcher fall outside the world size, or distributed training was
                launched on a device NCCL cannot use.
        """
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"LOCAL_RANK is {self.gpu_local_rank}, which is not below WORLD_SIZE {self.gpu_world_size}."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"RANK is {self.gpu_global_rank}, which is not below WORLD_SIZE {self.gpu_world_size}."
            )
        if not self.device.startswith("cuda"):
            raise ValueError(
                f"WORLD_SIZE is {self.gpu_world_size}, so this is a distributed run, but the device resolved to "
                f"'{self.device}'. Distributed training goes over NCCL, which needs CUDA. Launch on GPUs, or run a "
                "single process."
            )

        expected_device = f"cuda:{self.gpu_local_rank}"
        if self.device != expected_device:
            logger.warning(
                f"   Rank {self.gpu_global_rank} resolved to '{self.device}', but local rank "
                f"{self.gpu_local_rank} owns '{expected_device}'. Training on '{expected_device}'."
            )
            self.device = expected_device

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        # Guarded, because a second runner built in the same process would otherwise re-initialize a
        # group that is already up, which torch rejects
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(
                backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size
            )
        torch.cuda.set_device(self.gpu_local_rank)


# Runner classes that can be named by a runner config's class_name. A distillation runner, a
# multi-agent one, or any other OnPolicyRunner-shaped subclass registers here to become nameable
# the same way "PPO" names an algorithm or "MLPModel" names a model.
RUNNERS = {
    "OnPolicyRunner": OnPolicyRunner,
}


def create_runner(
    env: VecEnv,
    runner_cfg: OnPolicyRunnerConfig,
    device: str = "cpu",
) -> OnPolicyRunner:
    """Build the runner named by a configuration's ``class_name``.

    Calling ``OnPolicyRunner(...)`` directly always builds that one class; going through this
    function is what makes ``runner_cfg.class_name`` mean something, the same way an algorithm or a
    model config's ``class_name`` is resolved rather than read for documentation. A name is tried
    against :data:`RUNNERS` first and, if not found there, imported as a ``"module:Class"`` path, so
    a custom runner does not have to be registered to be used, only importable.

    Args:
        env: Vectorized environment for parallel experience collection.
        runner_cfg: Runner configuration. Its ``class_name`` selects which runner class to build.
        device: Device for computation ("cpu" or "cuda:0", etc.).

    Returns:
        The constructed runner.

    Raises:
        ImportError: If ``class_name`` is an import path and the module cannot be imported.
        AttributeError: If ``class_name`` is an import path and the module has no such attribute.
    """
    name = runner_cfg.class_name
    if callable(name):
        runner_class = name
    elif name in RUNNERS:
        runner_class = RUNNERS[name]
    else:
        runner_class = resolve_callable(name)
    return runner_class(env=env, runner_cfg=runner_cfg, device=device)
