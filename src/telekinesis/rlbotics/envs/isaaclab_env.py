"""Isaac Lab vector environment wrapper.

Isaac Lab runs Isaac Sim, simulating thousands of environments on an NVIDIA GPU. It needs Python
3.11, Linux (GLIBC 2.35+) or Windows and a GPU — there is no macOS build. Optional dependency::

    pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url https://pypi.nvidia.com
"""
from __future__ import annotations

import importlib.util

import torch
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.utils import resolve_device

# Checked by spec rather than by importing, because importing isaaclab pulls in Omniverse extensions
# that need the simulator app to be running
ISAACLAB_AVAILABLE = importlib.util.find_spec("isaaclab") is not None

INSTALL_HINT = (
    'Install it with: pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url '
    "https://pypi.nvidia.com (needs Python 3.11, Linux or Windows, and an NVIDIA GPU)"
)

# The running Isaac Sim app, if this process launched one
_SIMULATION_APP = None


def launch_simulator(headless: bool = True, **kwargs) -> object:
    """Start Isaac Sim, which has to happen before any Isaac Lab task is imported.

    Calling this more than once returns the app that is already running, since Isaac Sim is a
    process-wide singleton.

    Args:
        headless: Whether to run without a viewer window. Defaults to True, which is what training
            wants.
        **kwargs: Forwarded to Isaac Lab's ``AppLauncher``, for example ``enable_cameras=True``.

    Returns:
        The running simulation app.

    Raises:
        ImportError: If Isaac Lab is not installed.
    """
    global _SIMULATION_APP

    if _SIMULATION_APP is None:
        if not ISAACLAB_AVAILABLE:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"Launching Isaac Sim needs Isaac Lab, which is not installed. {INSTALL_HINT}"
            )

        from isaaclab.app import AppLauncher

        logger.info(f"   Launching Isaac Sim (headless={headless}). The first start takes a while.")
        _SIMULATION_APP = AppLauncher(headless=headless, **kwargs).app

    return _SIMULATION_APP


def shutdown_simulator() -> None:
    """Close Isaac Sim if this process started it.

    Isaac Sim does not exit with the environment, so a script that finishes without calling this can
    hang on shutdown.
    """
    global _SIMULATION_APP

    if _SIMULATION_APP is not None:
        _SIMULATION_APP.close()
        _SIMULATION_APP = None


def registered_tasks() -> list[str]:
    """Return the ids of every registered Isaac Lab task.

    Isaac Lab registers its tasks with Gymnasium and prefixes them with "Isaac-", so this is that
    registry filtered down. The simulator has to be running first, which this does.

    Returns:
        The task ids, sorted.

    Raises:
        ImportError: If Isaac Lab is not installed.
    """
    launch_simulator()
    gym, _ = _load_isaaclab()
    return sorted(task_id for task_id in gym.registry if task_id.startswith("Isaac-"))


def _load_isaaclab():
    """Import the Isaac Lab pieces this adapter needs.

    Deferred on purpose: these modules only import once Isaac Sim is running. Importing
    ``isaaclab_tasks`` is also what registers the tasks.

    Returns:
        Gymnasium and ``parse_env_cfg``.
    """
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    return gym, parse_env_cfg


class IsaacLabVecEnv(VecEnv):
    """Presents an Isaac Lab task as a :class:`VecEnv`.

    Everything about the task comes from its registered configuration: the observation groups, the
    number of actuators, the episode length and the simulation itself. Only the number of
    environments and the device are worth overriding from the outside.

    Isaac Lab names its observation groups "policy" and, for an asymmetric actor-critic, "critic".
    Actions are joint targets rather than a normalized range, so this adapter exposes no action
    bounds: a policy exported with
    :meth:`~telekinesis.rlbotics.runner.OnPolicyRunner.export` carries the observation normalization
    but no action scaling, and a deployment has to apply the same ``clip_actions``. Clipping happens
    here, in :meth:`step`, before the actions reach the task. Isaac Lab's own wrapper additionally
    rewrites the task's ``action_space`` to the clip range; that is skipped, because nothing in this
    library reads the space — the runner sizes the policy from ``num_actions``.

    Example:
        Train a locomotion task with this library's runner::

            env = IsaacLabVecEnv("Isaac-Velocity-Flat-Anymal-C-v0", num_envs=4096, device="cuda")
            runner = OnPolicyRunner(env=env, runner_cfg=cfg, device="cuda")
            ...
            env.close()
            shutdown_simulator()
    """

    def __init__(
        self,
        task: str,
        num_envs: int | None = None,
        device: str = "auto",
        clip_actions: float | None = None,
        headless: bool = True,
        render_mode: str | None = None,
    ) -> None:
        """Create the vector environment, launching Isaac Sim if it is not already running.

        Args:
            task: Registered Isaac Lab task id, such as "Isaac-Velocity-Flat-Anymal-C-v0". Use
                :func:`registered_tasks` to see what is available.
            num_envs: Number of parallel environments. Defaults to None, which keeps the number the
                task's own config asks for.
            device: Device the simulation and the observations live on. Defaults to "auto".
            clip_actions: Symmetric limit applied to the actions before they reach the task.
                Defaults to None, which passes them through.
            headless: Whether to run Isaac Sim without a viewer window. Defaults to True.
            render_mode: Passed straight to Isaac Lab. "rgb_array" starts the offscreen render
                pipeline so :meth:`render` returns frames of environment 0, which is what recording
                a video needs; this turns cameras on regardless of ``headless``. Defaults to None,
                which renders nothing.

        Raises:
            ImportError: If Isaac Lab is not installed.
            ValueError: If the task id is not registered.
        """
        launch_simulator(headless=headless, enable_cameras=render_mode is not None)
        gym, parse_env_cfg = _load_isaaclab()

        self.task = task
        self.clip_actions = clip_actions
        device = self._resolve_device(device)
        # parse_env_cfg applies the overrides to the task's registered config
        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)

        try:
            self.venv = gym.make(task, cfg=env_cfg, render_mode=render_mode)
        except gym.error.Error as error:
            raise ValueError(self._unknown_task_message(gym, task)) from error

        # Read back off the simulation, so the numbers are what Isaac Lab really built
        self.num_envs = self.unwrapped.num_envs
        self.num_actions = self._action_dim(gym)
        self.device = torch.device(self.unwrapped.device)
        self.max_episode_length = int(self.unwrapped.max_episode_length)
        # Isaac Lab does not reset when the task is constructed, and the first rollout step needs an
        # observation, so this is where the episode starts
        self.obs = self._observations(self.venv.reset()[0])

    @property
    def unwrapped(self):
        """Return the task underneath Gymnasium's wrappers, which is where its attributes live.

        ``gym.make`` hands back a wrapper chain, so the step and reset calls go through
        :attr:`venv` while everything else — the environment count, the device, the managers — is
        read from the ``ManagerBasedRLEnv`` or ``DirectRLEnv`` at the bottom of it.
        """
        return self.venv.unwrapped

    def _action_dim(self, gym) -> int:
        """Return the size of the action vector the task expects.

        Isaac Lab has two task workflows and they report this differently: a manager-based task owns
        an action manager that knows the total dimension, while a direct task only publishes an
        action space to measure.

        Args:
            gym: The Gymnasium module.

        Returns:
            The number of actions per environment.
        """
        manager = getattr(self.unwrapped, "action_manager", None)
        if manager is not None:
            return int(manager.total_action_dim)
        return int(gym.spaces.flatdim(self.unwrapped.single_action_space))

    def _observations(self, obs: dict[str, torch.Tensor]) -> TensorDict:
        """Wrap Isaac Lab's observation groups in a TensorDict.

        Args:
            obs: Observations, keyed by group name.

        Returns:
            The observations, batched over environments.
        """
        return TensorDict(obs, batch_size=(self.num_envs,))

    @staticmethod
    def _unknown_task_message(gym, task: str) -> str:
        """Build the error message for a task id that is not registered.

        Args:
            gym: The Gymnasium module.
            task: The task id that was asked for.

        Returns:
            A message naming a few of the registered Isaac Lab tasks.
        """
        available = sorted(task_id for task_id in gym.registry if task_id.startswith("Isaac-"))
        examples = ", ".join(available[:5]) if available else "none"
        return (
            f"'{task}' is not a registered Isaac Lab task. {len(available)} are registered, "
            f"for example: {examples}. Call "
            "telekinesis.rlbotics.envs.isaaclab_env.registered_tasks() for the full list."
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve the device, warning when it is one Isaac Lab cannot train on.

        Args:
            device: Requested device, or "auto".

        Returns:
            A device string Isaac Lab accepts.
        """
        resolved = resolve_device(device)
        if resolved.startswith("mps"):
            logger.warning(
                "   Isaac Sim has no Apple Silicon build, so 'mps' is not a device it can use. "
                "Falling back to cpu, which needs an NVIDIA GPU present anyway."
            )
            return "cpu"
        if not resolved.startswith("cuda"):
            logger.warning(
                "   Isaac Lab on cpu runs the physics pipeline on the host and is far slower. "
                "Keep num_envs small."
            )
        return resolved

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return Isaac Lab's own per-environment step counter.

        It is delegated rather than duplicated so that writing to it, which is how the runner
        staggers episode lengths at the start of training, reaches the simulation.
        """
        return self.unwrapped.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Write Isaac Lab's per-environment step counter."""
        self.unwrapped.episode_length_buf = value

    @property
    def cfg(self):
        """Return the task's own configuration, read live off the simulation.

        A property rather than a value copied in at construction, so it stays correct even if
        something else mutates the task's config after this adapter is built.
        """
        return self.unwrapped.cfg

    def get_observations(self) -> TensorDict:
        """Return fresh observations without stepping, recomputed rather than replayed.

        Isaac Lab's own wrapper does this too: whatever :meth:`step` or :meth:`reset` last cached
        only reflects the state at that call, so a caller in between — logging, or an algorithm
        peeking before it acts — would otherwise see a stale observation if anything else touched
        the simulation. The two task workflows publish it differently, the same way
        :meth:`_action_dim` reads two different places for the action count: a manager-based task
        through its observation manager, a direct task through its own method.
        """
        manager = getattr(self.unwrapped, "observation_manager", None)
        obs = manager.compute() if manager is not None else self.unwrapped._get_observations()
        return self._observations(obs)

    def seed(self, seed: int = -1) -> int:
        """Reseed the task's own random number generator.

        This is separate from :func:`~telekinesis.rlbotics.utils.set_seed`, which seeds model
        init, action sampling and mini-batch order on the training side: the simulation has its
        own domain randomization and reset noise, which lives here instead.

        Args:
            seed: Seed to use. Defaults to -1, which asks Isaac Lab to pick one.

        Returns:
            The seed that was actually used.
        """
        return self.unwrapped.seed(seed)

    def reset(self) -> TensorDict:
        """Reset all environments."""
        self.obs = self._observations(self.venv.reset()[0])
        return self.obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments.

        Args:
            actions: Policy actions, one row per environment, clipped to ``clip_actions`` first when
                one was given.

        Returns:
            Observations, rewards and dones of shape (num_envs,), and Isaac Lab's extras. A task
            with an infinite horizon carries the truncation flags under "time_outs" so the
            algorithm can bootstrap episodes cut off by the time limit; a finite-horizon task does
            not, because there the limit is part of the task.
        """
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)

        obs, rewards, terminated, truncated, extras = self.venv.step(actions)
        self.obs = self._observations(obs)

        # Isaac Lab keeps termination and truncation apart, while the contract wants one done flag
        # that can be used arithmetically, plus the truncations for bootstrapping
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self.obs, rewards, (terminated | truncated).to(dtype=torch.float32), extras

    def close(self) -> None:
        """Close the task. Isaac Sim keeps running, see :func:`shutdown_simulator`."""
        self.venv.close()

    def render(self):
        """Return the current frame as an RGB array.

        Only works when the environment was built with ``render_mode="rgb_array"``; otherwise
        Isaac Lab returns None.

        Returns:
            An HxWx3 ``uint8`` array, or None if rendering was not requested.
        """
        return self.unwrapped.render()

    @property
    def render_fps(self) -> float:
        """Return the simulation's step rate, for timing a recorded video."""
        return self.unwrapped.metadata.get("render_fps", 30.0)
