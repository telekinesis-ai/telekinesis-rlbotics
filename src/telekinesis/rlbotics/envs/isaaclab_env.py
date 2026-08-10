"""Isaac Lab vector environment wrapper.

Isaac Lab runs Isaac Sim, so it simulates thousands of environments on an NVIDIA GPU with rendering,
sensors and a large task library. It is an optional dependency::

    pip install "isaaclab[isaacsim,all]>=2.3" --extra-index-url https://pypi.nvidia.com

That extra index is required, since the Isaac Sim wheels are hosted by NVIDIA rather than PyPI.
Isaac Lab needs Python 3.11, Linux (GLIBC 2.35+) or Windows, and an NVIDIA GPU. There is no macOS
build, so unlike the Gymnasium and mjlab adapters this one cannot be run on a laptop at all.

One thing is different here from every other adapter: Isaac Sim has to be running before any task
module is imported, because those modules import Omniverse extensions that only exist once the app
is up. So the imports are deferred until :func:`launch_simulator` has run, which
:class:`IsaacLabVecEnv` does for you. Build the environment before importing anything from
``isaaclab_tasks`` yourself.
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
    'Install it with: pip install "isaaclab[isaacsim,all]>=2.3" --extra-index-url '
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
    gym, _, _ = _load_isaaclab()
    return sorted(task_id for task_id in gym.registry if task_id.startswith("Isaac-"))


def _load_isaaclab():
    """Import the Isaac Lab pieces this adapter needs.

    Deferred on purpose: these modules only import once Isaac Sim is running. Importing
    ``isaaclab_tasks`` is also what registers the tasks.

    Returns:
        Gymnasium, ``parse_env_cfg`` and Isaac Lab's rsl_rl vector environment wrapper.
    """
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    return gym, parse_env_cfg, RslRlVecEnvWrapper


class IsaacLabVecEnv(VecEnv):
    """Presents an Isaac Lab task as a :class:`VecEnv`.

    Everything about the task comes from its registered configuration: the observation groups, the
    number of actuators, the episode length and the simulation itself. Only the number of
    environments and the device are worth overriding from the outside.

    Isaac Lab names its observation groups "policy" and, for an asymmetric actor-critic, "critic".
    Actions are joint targets rather than a normalized range, so this adapter exposes no action
    bounds: a policy exported with
    :meth:`~telekinesis.rlbotics.runner.OnPolicyRunner.export` carries the observation normalization
    but no action scaling, and a deployment has to apply the same ``clip_actions``.

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
        gym, parse_env_cfg, RslRlVecEnvWrapper = _load_isaaclab()

        self.task = task
        device = self._resolve_device(device)
        # parse_env_cfg applies the overrides to the task's registered config
        env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)

        try:
            env = gym.make(task, cfg=env_cfg, render_mode=render_mode)
        except gym.error.Error as error:
            raise ValueError(self._unknown_task_message(gym, task)) from error

        self.venv = RslRlVecEnvWrapper(env, clip_actions=clip_actions)
        # The wrapper reads the numbers back off the simulation, so they are what Isaac Lab built
        self.num_envs = self.venv.num_envs
        self.num_actions = self.venv.num_actions
        self.device = torch.device(self.venv.device)
        self.max_episode_length = int(self.venv.max_episode_length)
        self.obs = self.venv.get_observations()

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
        return self.venv.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Write Isaac Lab's per-environment step counter."""
        self.venv.episode_length_buf = value

    def get_observations(self) -> TensorDict:
        """Return the current observations without stepping."""
        return self.obs

    def reset(self) -> TensorDict:
        """Reset all environments."""
        self.obs, _ = self.venv.reset()
        return self.obs

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments.

        Args:
            actions: Policy actions, one row per environment.

        Returns:
            Observations, rewards and dones of shape (num_envs,), and Isaac Lab's extras. A task
            with an infinite horizon carries the truncation flags under "time_outs" so the
            algorithm can bootstrap episodes cut off by the time limit; a finite-horizon task does
            not, because there the limit is part of the task.
        """
        self.obs, rewards, dones, extras = self.venv.step(actions)
        # Isaac Lab reports dones as integers, while the contract and the rollout buffer want flags
        # that can be used arithmetically
        return self.obs, rewards, dones.to(dtype=torch.float32), extras

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
        return self.venv.unwrapped.render()

    @property
    def render_fps(self) -> float:
        """Return the simulation's step rate, for timing a recorded video."""
        return self.venv.unwrapped.metadata.get("render_fps", 30.0)
