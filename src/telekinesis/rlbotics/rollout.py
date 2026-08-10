"""Rollout storage for collecting experience during environment interactions."""

from __future__ import annotations

from collections.abc import Generator

import torch


class Transition:
    """Storage for a single state transition.

    This class is populated incrementally during the rollout phase and then passed to
    RolloutBuffer.add_transition to record the data.
    """

    def __init__(self) -> None:
        """Initialize an empty transition container."""
        # Observations
        self.observations: torch.Tensor | None = None
        """Observations at the current step, as consumed by the actor."""

        self.critic_observations: torch.Tensor | None = None
        """Observations at the current step, as consumed by the critic.

        ``None`` if identical to the actor's.
        """

        self.actions: torch.Tensor | None = None
        """Actions taken at the current step."""

        self.rewards: torch.Tensor | None = None
        """Rewards received after the action."""

        self.dones: torch.Tensor | None = None
        """Done flags indicating episode termination."""

        # For reinforcement learning
        self.values: torch.Tensor | None = None
        """Value estimates at the current step (RL only)."""

        self.actions_log_prob: torch.Tensor | None = None
        """Log probability of the taken actions (RL only)."""

        self.distribution_params: tuple[torch.Tensor, ...] | None = None
        """Parameters of the action distribution (RL only)."""

        # For distillation
        self.privileged_actions: torch.Tensor | None = None
        """Privileged (teacher) actions (distillation only)."""

        # For recurrent networks
        self.hidden_states: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None)
        """Hidden states for recurrent networks, e.g., (actor, critic)."""

    def clear(self) -> None:
        """Reset all transition fields to None."""
        self.__init__()


class RolloutBufferBatch:
    """A batch of data yielded by the rollout buffer generators.

    This class provides named access to mini-batch fields. Fields are optional to support
    different training modes (RL vs distillation) and architectures (feedforward vs recurrent).
    """

    def __init__(
        self,
        observations: torch.Tensor | None = None,
        critic_observations: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        values: torch.Tensor | None = None,
        advantages: torch.Tensor | None = None,
        returns: torch.Tensor | None = None,
        old_actions_log_prob: torch.Tensor | None = None,
        old_distribution_params: tuple[torch.Tensor, ...] | None = None,
        hidden_states: tuple[torch.Tensor | None, torch.Tensor | None] = (None, None),
        masks: torch.Tensor | None = None,
        privileged_actions: torch.Tensor | None = None,
        dones: torch.Tensor | None = None,
    ) -> None:
        """Initialize a batch container over rollout data."""
        self.observations: torch.Tensor | None = observations
        """Batch of observations, as consumed by the actor."""

        self.critic_observations: torch.Tensor | None = critic_observations
        """Batch of observations, as consumed by the critic.

        Falls back to ``observations`` when not provided.
        """

        # For reinforcement learning
        self.actions: torch.Tensor | None = actions
        """Batch of actions."""

        self.values: torch.Tensor | None = values
        """Batch of value estimates (RL only)."""

        self.advantages: torch.Tensor | None = advantages
        """Batch of advantage estimates (RL only)."""

        self.returns: torch.Tensor | None = returns
        """Batch of return targets (RL only)."""

        self.old_actions_log_prob: torch.Tensor | None = old_actions_log_prob
        """Batch of log probabilities of the old actions (RL only)."""

        self.old_distribution_params: tuple[torch.Tensor, ...] | None = old_distribution_params
        """Batch of parameters of the old action distribution (RL only)."""

        # For distillation
        self.privileged_actions: torch.Tensor | None = privileged_actions
        """Batch of privileged (teacher) actions (distillation only)."""

        self.dones: torch.Tensor | None = dones
        """Batch of done flags (distillation only)."""

        # For recurrent networks
        self.hidden_states: tuple[torch.Tensor | None, torch.Tensor | None] = hidden_states
        """Batch of hidden states for recurrent networks (RL recurrent only)."""

        self.masks: torch.Tensor | None = masks
        """Batch of trajectory masks for recurrent networks (RL recurrent only)."""


class RolloutBuffer:
    """Storage for data collected during a rollout.

    The rollout buffer is populated by adding transitions during the rollout phase.
    It then returns a generator for learning, depending on the algorithm and policy architecture.
    """

    def __init__(
        self,
        training_type: str,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple[int, ...] | list[int],
        actions_shape: tuple[int, ...] | list[int],
        device: str = "cpu",
        critic_obs_shape: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Allocate rollout buffers for a specific training mode and batch shape.

        Args:
            training_type: Type of training ("rl" or "distillation").
            num_envs: Number of parallel environments.
            num_transitions_per_env: Number of transitions per environment per rollout.
            obs_shape: Shape of the actor observations (excluding batch dimensions).
            actions_shape: Shape of actions (excluding batch dimensions).
            device: Device to store tensors on ("cpu" or "cuda").
            critic_obs_shape: Shape of the critic observations (excluding batch dimensions).
                Defaults to None, in which case the critic shares the actor's observation buffer.
        """
        self.training_type = training_type
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.actions_shape = actions_shape
        self.obs_shape = obs_shape

        # Core buffers
        self.observations = torch.zeros(
            num_transitions_per_env, num_envs, *obs_shape, device=device
        )

        # The critic observes a different set only when privileged observations are configured, so
        # the buffer is shared
        # whenever the two sets match
        self.has_critic_obs = critic_obs_shape is not None and tuple(critic_obs_shape) != tuple(
            obs_shape
        )
        self.critic_obs_shape = tuple(critic_obs_shape) if self.has_critic_obs else tuple(obs_shape)
        if self.has_critic_obs:
            self.critic_observations = torch.zeros(
                num_transitions_per_env, num_envs, *self.critic_obs_shape, device=device
            )
        else:
            self.critic_observations = self.observations

        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()

        # For distillation
        if training_type == "distillation":
            self.privileged_actions = torch.zeros(
                num_transitions_per_env, num_envs, *actions_shape, device=device
            )

        # For reinforcement learning
        if training_type == "rl":
            self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
            self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
            self.distribution_params: tuple[torch.Tensor, ...] | None = None
            self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
            self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)

        # For recurrent networks
        self.saved_hidden_state_actor = None
        self.saved_hidden_state_critic = None

        # Counter for the number of transitions stored
        self.step = 0

    def add_transition(self, transition: Transition) -> None:
        """Add one transition to the buffer at the current step index.

        Args:
            transition: A Transition object containing the data to add.

        Raises:
            OverflowError: If the buffer is full.
        """
        if self.step >= self.num_transitions_per_env:
            raise OverflowError(
                "Rollout buffer overflow! You should call clear() before adding new transitions."
            )

        # Core buffers
        self.observations[self.step].copy_(transition.observations)
        if self.has_critic_obs:
            if transition.critic_observations is None:
                raise ValueError(
                    "The buffer was allocated with a separate critic observation shape, so every "
                    "transition must"
                    " provide 'critic_observations'."
                )
            self.critic_observations[self.step].copy_(transition.critic_observations)
        if transition.actions is not None:
            self.actions[self.step].copy_(transition.actions)
        if transition.rewards is not None:
            self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        if transition.dones is not None:
            self.dones[self.step].copy_(transition.dones.view(-1, 1))

        # For distillation
        if self.training_type == "distillation":
            self.privileged_actions[self.step].copy_(transition.privileged_actions)

        # For reinforcement learning
        if self.training_type == "rl":
            self.values[self.step].copy_(transition.values)
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))

            # Initialize distribution parameters on first transition
            if self.distribution_params is None:
                self.distribution_params = tuple(
                    torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                    for p in transition.distribution_params
                )

            # Copy distribution parameters
            for i, p in enumerate(transition.distribution_params):
                self.distribution_params[i][self.step].copy_(p)

        # Save hidden states for recurrent networks
        self._save_hidden_states(transition.hidden_states)

        self.step += 1

    def clear(self) -> None:
        """Reset the write cursor for the next rollout."""
        self.step = 0

    def compute_returns_and_advantages(
        self, last_values: torch.Tensor, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> None:
        """Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Args:
            last_values: Value estimates for the state after the last transition.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.

        Raises:
            ValueError: If not in RL mode.
        """
        if self.training_type != "rl":
            raise ValueError("compute_returns_and_advantages is only available for RL training.")

        advantages = torch.zeros_like(self.rewards)
        next_value = last_values.clone()

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = last_values
            else:
                next_non_terminal = 1.0 - self.dones[step]
                next_value = self.values[step + 1]

            delta = self.rewards[step] + gamma * next_value * next_non_terminal - self.values[step]

            advantages[step] = delta
            if step < self.num_transitions_per_env - 1:
                advantages[step] += gamma * gae_lambda * next_non_terminal * advantages[step + 1]

        self.returns = advantages + self.values
        self.advantages = advantages

        # Normalize advantages
        advantages_mean = self.advantages.mean()
        advantages_std = self.advantages.std()
        self.advantages = (self.advantages - advantages_mean) / (advantages_std + 1e-8)

    def generator(self) -> Generator[RolloutBufferBatch, None, None]:
        """Yield per-timestep batches for distillation training.

        Yields:
            RolloutBufferBatch: A batch containing observations and privileged actions.

        Raises:
            ValueError: If not in distillation mode.
        """
        if self.training_type != "distillation":
            raise ValueError("This function is only available for distillation training.")

        for i in range(self.num_transitions_per_env):
            yield RolloutBufferBatch(
                observations=self.observations[i],
                privileged_actions=self.privileged_actions[i],
                dones=self.dones[i],
            )

    def mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutBufferBatch, None, None]:
        """Yield shuffled flat mini-batches for feedforward RL updates.

        Args:
            num_mini_batches: Number of mini-batches to split the data into.
            num_epochs: Number of epochs to iterate through the data.

        Yields:
            RolloutBufferBatch: A mini-batch containing all RL fields.

        Raises:
            ValueError: If not in RL mode.
        """
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")

        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(
            num_mini_batches * mini_batch_size,
            requires_grad=False,
            device=self.device,
        )

        # Flatten the data
        observations = self.observations.reshape(-1, *self.obs_shape)
        critic_observations = (
            self.critic_observations.reshape(-1, *self.critic_obs_shape)
            if self.has_critic_obs
            else observations
        )
        actions = self.actions.reshape(-1, *self.actions_shape)
        values = self.values.reshape(-1, 1)
        returns = self.returns.reshape(-1, 1)
        old_actions_log_prob = self.actions_log_prob.reshape(-1, 1)
        advantages = self.advantages.reshape(-1, 1)
        old_distribution_params = tuple(
            p.reshape(-1, *p.shape[2:]) for p in self.distribution_params
        )

        for _epoch in range(num_epochs):
            for i in range(num_mini_batches):
                # Select the indices for the mini-batch
                start = i * mini_batch_size
                stop = (i + 1) * mini_batch_size
                batch_idx = indices[start:stop]

                yield RolloutBufferBatch(
                    observations=observations[batch_idx],
                    critic_observations=critic_observations[batch_idx],
                    actions=actions[batch_idx],
                    values=values[batch_idx],
                    advantages=advantages[batch_idx],
                    returns=returns[batch_idx],
                    old_actions_log_prob=old_actions_log_prob[batch_idx],
                    old_distribution_params=tuple(p[batch_idx] for p in old_distribution_params),
                )

    def _save_hidden_states(
        self, hidden_states: tuple[torch.Tensor | None, torch.Tensor | None]
    ) -> None:
        """Save recurrent hidden states to the rollout buffer.

        Args:
            hidden_states: Tuple of (actor_hidden_state, critic_hidden_state).
        """
        if hidden_states == (None, None):
            return

        # Initialize hidden states if needed
        if self.saved_hidden_state_actor is None and hidden_states[0] is not None:
            self.saved_hidden_state_actor = torch.zeros(
                self.observations.shape[0],
                *hidden_states[0].shape,
                device=self.device,
            )
        if self.saved_hidden_state_critic is None and hidden_states[1] is not None:
            self.saved_hidden_state_critic = torch.zeros(
                self.observations.shape[0],
                *hidden_states[1].shape,
                device=self.device,
            )

        # Copy the states
        if hidden_states[0] is not None:
            self.saved_hidden_state_actor[self.step].copy_(hidden_states[0])
        if hidden_states[1] is not None:
            self.saved_hidden_state_critic[self.step].copy_(hidden_states[1])
