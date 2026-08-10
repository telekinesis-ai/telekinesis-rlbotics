"""Vectorized environment interface for reinforcement learning."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from tensordict import TensorDict


class VecEnv(ABC):
    """The interface a vectorized environment satisfies to be trained on.

    A vectorized environment steps many environments in parallel, applying one batch of actions and
    returning batched observations, rewards and dones.

    This is an interface, not a base class that holds state: it has no ``__init__``, and the
    attributes below are declarations of what the runner and the algorithm read, to be set by the
    environment itself. That matters because an environment usually does not own these numbers — it
    reads them off the simulator it wraps, and any buffer belongs to whoever owns the episode. A base
    class that allocated them would either duplicate the simulator's state or, worse, overwrite it.
    """

    num_envs: int
    """Number of environments stepped in parallel."""

    num_actions: int
    """Size of the action vector one environment expects."""

    device: torch.device | str
    """Device the observations, rewards and dones are placed on."""

    max_episode_length: int | torch.Tensor
    """Steps an episode runs before the time limit cuts it off.

    A scalar applies to every environment; a tensor gives a limit per environment.
    """

    episode_length_buf: torch.Tensor
    """How many steps each environment is into its episode.

    Optional. Only an environment that supports ``learn(init_at_random_ep_len=True)`` needs it, and
    for a simulator-backed environment it belongs to the simulator, so expose it as a property that
    reads and writes through rather than as a buffer of your own.
    """

    cfg: dict | object
    """The environment's own configuration, for logging and introspection."""

    @abstractmethod
    def reset(self) -> TensorDict:
        """Reset all environments.

        Returns:
            TensorDict containing observations from all environments after reset.
        """
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments.

        Args:
            actions: Tensor of shape (num_envs, num_actions) containing actions for each
                environment.

        Returns:
            observations: TensorDict with observations from all environments.
            rewards: Tensor of shape (num_envs,) containing rewards.
            dones: Tensor of shape (num_envs,) with done flags (True if episode terminated).
            extras: Dictionary with extra information including:
                - "time_outs": Tensor of shape (num_envs,) indicating time limit terminations.
                - "log": Dictionary with logging information.
        """
        raise NotImplementedError

    @abstractmethod
    def get_observations(self) -> TensorDict:
        """Return current observations without stepping.

        Returns:
            TensorDict containing current observations.
        """
        raise NotImplementedError


def observation_spec(env: VecEnv) -> dict[str, tuple[int, ...]]:
    """Return the per-environment shape of each observation group.

    This is a function rather than part of :class:`VecEnv` because it is derived, not declared: the
    runner sizes its models from the observations themselves, so nothing in the library needs an
    environment to describe them in advance. It is here for scripts that have to know what a task
    publishes before they can configure ``obs_groups``.

    Args:
        env: The environment to inspect.

    Returns:
        Each observation group's shape, without the environment dimension.
    """
    return {group: tuple(value.shape[1:]) for group, value in env.get_observations().items()}
