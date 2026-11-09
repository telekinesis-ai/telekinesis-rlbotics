"""Gymnasium vector environment wrapper.

Gymnasium is an optional dependency: the library trains against any :class:`VecEnv`, and this module
is only the adapter for Gymnasium's tasks. Install it with::

    pip install "telekinesis-rlbotics[gym]"
"""
from __future__ import annotations

import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.utils import resolve_device

try:
    import gymnasium as gym

    GYMNASIUM_IMPORT_ERROR: ImportError | None = None
except ImportError as error:  # pragma: no cover - depends on the environment
    gym = None
    GYMNASIUM_IMPORT_ERROR = error

INSTALL_HINT = 'Install it with: pip install "telekinesis-rlbotics[gym]"'

EPISODE_METRICS_KEY = "episode_metrics"
"""The ``infos`` entry a task reports per-episode metrics under, forwarded as ``extras["episode"]``.

Its value is a dictionary of per-environment arrays, read only where an episode ends (a success
flag, a final distance). Gymnasium's vectorizers merge each sub-environment's dictionary into this
shape on their own, adding a ``_<key>`` mask for the environments that reported the key, and a
custom vector environment builds it directly.
"""


class GymnasiumVecEnv(VecEnv):
    """Presents a Gymnasium vector environment as a :class:`VecEnv`.

    Converts between numpy and torch, and maps policy actions in ``[-1, 1]`` onto the environment's
    action bounds. Everything else is read from the environment id, so any continuous control task
    works by name: the observation size, the number of actuators, the action bounds and the episode
    limit all come from the spec.

    Example:
        Wrap a task and check what was read from it::

            env = GymnasiumVecEnv("Hopper-v5", num_envs=32, device="auto")
            env.obs_dim, env.num_actions, env.max_episode_length  # (11, 3, 1000)
    """

    def __init__(
        self, env_id: str, num_envs: int, device: str, render_mode: str | None = None
    ) -> None:
        """Create the vector environment.

        Args:
            env_id: Gymnasium environment id with a continuous (Box) action space and a flat
                observation, which covers the classic control and MuJoCo tasks.
            num_envs: Number of parallel environments.
            device: Device the observations and rewards are placed on.
            render_mode: Passed straight to Gymnasium. "rgb_array" makes :meth:`render` return
                frames, which is what recording a video needs. Defaults to None, which renders
                nothing.

        Raises:
            ImportError: If Gymnasium, or the extra a task needs, is not installed.
            ValueError: If the task's action space is not continuous, or its observation is not flat.
        """
        if gym is None:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"GymnasiumVecEnv needs Gymnasium, which is not installed. {INSTALL_HINT}"
            ) from GYMNASIUM_IMPORT_ERROR

        try:
            self.venv = self._make_vec(env_id, num_envs, render_mode)
        except gym.error.DependencyNotInstalled as error:
            # MuJoCo and Box2D tasks each pull in their own physics engine
            raise ImportError(
                f"'{env_id}' needs a Gymnasium extra that is not installed: {error} {INSTALL_HINT}"
            ) from error

        action_space = self.venv.single_action_space
        obs_space = self.venv.single_observation_space
        self._check_spaces(env_id, action_space, obs_space)

        self.num_envs = num_envs
        self.num_actions = int(np.prod(action_space.shape))
        self.device = torch.device(resolve_device(device))

        self.obs_dim = int(obs_space.shape[0])
        self.max_episode_length = gym.spec(env_id).max_episode_steps or 1000
        self.action_low, self.action_high = self._action_bounds(env_id, action_space)
        self.obs = self._to_tensor(self.venv.reset(seed=0)[0])

    @staticmethod
    def _make_vec(env_id: str, num_envs: int, render_mode: str | None) -> gym.vector.VectorEnv:
        """Build the vector environment, natively batched if the task registers a vector entry point.

        A task that simulates its environments as one batch (a GPU or multithreaded simulator)
        registers ``vector_entry_point``, and is built through it rather than as ``num_envs`` copies
        stepped one by one. Every other task is wrapped in a synchronous vector environment.

        Either way the environments must autoreset on the same step they end, which keeps every
        step a real transition, as the rollout expects. Gymnasium accepts no ``vector_kwargs`` for
        a vector entry point, so such a task has to declare that mode in its metadata itself. One
        that does not (Gymnasium's own CartPole, say, which resets on the next step) is built
        synchronously from its single-environment entry point instead.

        Args:
            env_id: Gymnasium environment id.
            num_envs: Number of parallel environments.
            render_mode: Passed straight to Gymnasium, and only when set, so a vector entry point
                that does not render need not accept it.

        Returns:
            The vector environment.

        Raises:
            ValueError: If a task registers only a vector entry point, and it does not autoreset
                on the same step.
        """
        render = {} if render_mode is None else {"render_mode": render_mode}
        spec = gym.spec(env_id)
        if spec.vector_entry_point is not None:
            venv = gym.make_vec(
                env_id, num_envs=num_envs, vectorization_mode="vector_entry_point", **render
            )
            mode = venv.metadata.get("autoreset_mode")
            if mode == gym.vector.AutoresetMode.SAME_STEP:
                return venv
            venv.close()
            if spec.entry_point is None:
                raise ValueError(
                    f"'{env_id}' is a vector environment with autoreset mode {mode}, and only "
                    "SAME_STEP is supported: the rollout expects every step to be a real "
                    'transition. Declare metadata["autoreset_mode"] = '
                    "gym.vector.AutoresetMode.SAME_STEP."
                )

        return gym.make_vec(
            env_id,
            num_envs=num_envs,
            vectorization_mode="sync",
            vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP},
            **render,
        )

    @staticmethod
    def _check_spaces(env_id: str, action_space: gym.Space, obs_space: gym.Space) -> None:
        """Reject the task early if its spaces are not the ones this wrapper handles.

        Args:
            env_id: Gymnasium environment id, named in the error messages.
            action_space: The task's single-environment action space.
            obs_space: The task's single-environment observation space.

        Raises:
            ValueError: If the action space is not a continuous Box, or the observation is not a
                flat Box.
        """
        if not isinstance(action_space, gym.spaces.Box):
            raise ValueError(
                f"'{env_id}' has a {action_space} action space, and only continuous (Box) action "
                "spaces are supported: the policy is Gaussian, so it produces real-valued actions. "
                "Pick a continuous task, such as Pendulum-v1, MountainCarContinuous-v0 or any of "
                "the MuJoCo tasks (Hopper-v5, Walker2d-v5, Humanoid-v5)."
            )

        if not isinstance(obs_space, gym.spaces.Box) or len(obs_space.shape) != 1:
            raise ValueError(
                f"'{env_id}' has a {obs_space} observation space, and only flat Box observations "
                "are supported. Images need a CNN model and an adapter that keeps the observation "
                "shape; dictionary observations need one entry per observation group."
            )

    def _action_bounds(
        self, env_id: str, action_space: gym.spaces.Box
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Read the action bounds the policy output is mapped onto.

        Args:
            env_id: Gymnasium environment id, named in the warning.
            action_space: The task's single-environment action space.

        Returns:
            The low and high bounds as tensors on the device, or ``(None, None)`` if the task does
            not bound its actions, in which case actions are passed through unscaled.
        """
        low, high = action_space.low, action_space.high
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            logger.warning(
                f"   '{env_id}' does not bound its actions, so they are passed through unscaled. "
                "Watch Policy/action_std, since nothing limits how large an action can get."
            )
            return None, None

        return (
            torch.as_tensor(low, dtype=torch.float32, device=self.device),
            torch.as_tensor(high, dtype=torch.float32, device=self.device),
        )

    def _to_tensor(self, values: np.ndarray) -> torch.Tensor:
        """Convert numpy output to a float32 tensor on the device."""
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _observations(self) -> TensorDict:
        """Wrap the cached observations in a TensorDict."""
        return TensorDict(
            {"observation": self.obs}, batch_size=(self.num_envs,), device=self.device
        )

    @property
    def cfg(self):
        """Return the task's registration spec, read live off the vector environment.

        Gymnasium has no rich per-task configuration object the way the manager-based simulators
        do, so this is the closest equivalent for logging and introspection: the id, the entry
        point and whatever kwargs built it.
        """
        return self.venv.spec

    def get_observations(self) -> TensorDict:
        """Return the current observations without stepping.

        Unlike the manager-based adapters, this is necessarily the last cached observation:
        Gymnasium has no "compute the current observation" call independent of reset() or step(),
        so there is nothing fresher to recompute it from.
        """
        return self._observations()

    def seed(self, seed: int = -1) -> int:
        """Reseed the task's own random number generator.

        This is separate from :func:`~telekinesis.rlbotics.utils.set_seed`, which seeds model
        init, action sampling and mini-batch order on the training side: the simulation has its own
        reset randomness, which lives here instead. Gymnasium folds seeding into reset() itself
        rather than exposing a bare reseed call, so this necessarily starts a fresh episode too.

        Args:
            seed: Seed to use. Defaults to -1, which asks NumPy to pick one.

        Returns:
            The seed that was actually used.
        """
        if seed < 0:
            seed = int(np.random.randint(0, 2**31 - 1))
        self.obs = self._to_tensor(self.venv.reset(seed=seed)[0])
        return seed

    def reset(self) -> TensorDict:
        """Reset all environments."""
        self.obs = self._to_tensor(self.venv.reset()[0])
        return self._observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments.

        Args:
            actions: Policy actions, clipped and scaled onto the action bounds before stepping.

        Returns:
            Observations, rewards and dones of shape (num_envs,), and extras carrying the truncation
            flags under "time_outs" so the algorithm can bootstrap episodes cut off by the limit.
        """
        if self.action_low is None:
            scaled = actions
        else:
            scaled = self.action_low + 0.5 * (actions.clamp(-1.0, 1.0) + 1.0) * (
                self.action_high - self.action_low
            )
        obs, rewards, terminated, truncated, infos = self.venv.step(scaled.detach().cpu().numpy())

        self.obs = self._to_tensor(obs)
        extras = {"time_outs": torch.as_tensor(truncated, dtype=torch.bool, device=self.device)}
        metrics = self._episode_metrics(infos)
        if metrics:
            extras["episode"] = metrics
        return (
            self._observations(),
            self._to_tensor(rewards),
            self._to_tensor(terminated | truncated),
            extras,
        )

    def _episode_metrics(self, infos: dict) -> dict[str, torch.Tensor]:
        """Read the task's per-episode metrics out of a step's ``infos``.

        Only the :data:`EPISODE_METRICS_KEY` entry is read, so a task decides what is a metric
        rather than every number it puts in ``infos`` (a MuJoCo task's per-step reward terms, say)
        being logged as one.

        Args:
            infos: The vector environment's ``infos`` for this step.

        Returns:
            One float tensor of shape (num_envs,) per numeric key. An environment that did not
            report a key on this step holds NaN, which the logger leaves out of the mean.
        """
        reported = infos.get(EPISODE_METRICS_KEY)
        if not reported:
            return {}

        metrics = {}
        for key, values in reported.items():
            values = np.asarray(values)
            if key.startswith("_") or values.shape != (self.num_envs,) or values.dtype.kind not in "biuf":
                continue
            values = values.astype(np.float32)
            mask = reported.get(f"_{key}")
            if mask is not None:
                values = np.where(np.asarray(mask, dtype=bool), values, np.nan)
            metrics[key] = self._to_tensor(values)
        return metrics

    def render(self):
        """Return the current frame of each sub-environment.

        Only works when the environment was built with ``render_mode="rgb_array"``; otherwise
        Gymnasium returns None.

        Returns:
            A tuple of HxWx3 ``uint8`` arrays, one per sub-environment, or None if rendering was not
            requested.
        """
        return self.venv.render()

    @property
    def render_fps(self) -> float:
        """Return the frame rate the task renders at, for timing a recorded video."""
        return float(self.venv.metadata.get("render_fps", 30.0))

    def close(self) -> None:
        """Close the underlying environment."""
        self.venv.close()
