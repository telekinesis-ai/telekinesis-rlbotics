"""mjlab vector environment wrapper.

mjlab pairs Isaac Lab's manager-based API with MuJoCo Warp, so one task runs thousands of
environments on a GPU. Optional dependency::

    pip install "telekinesis-rlbotics[mjlab]"
"""
from __future__ import annotations

import re

import torch
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.utils import resolve_device

try:
    # Importing the task package is what populates mjlab's registry
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import list_tasks, load_env_cfg

    MJLAB_IMPORT_ERROR: ImportError | None = None
except ImportError as error:  # pragma: no cover - depends on the environment
    ManagerBasedRlEnv = None
    list_tasks = None
    load_env_cfg = None
    MJLAB_IMPORT_ERROR = error

INSTALL_HINT = 'Install it with: pip install "telekinesis-rlbotics[mjlab]"'


def registered_tasks() -> list[str]:
    """Return the ids of every mjlab task that is registered.

    Returns:
        The task ids, sorted.

    Raises:
        ImportError: If mjlab is not installed.
    """
    if list_tasks is None:  # pragma: no cover - depends on the environment
        raise ImportError(
            f"Listing mjlab tasks needs mjlab. {INSTALL_HINT}"
        ) from MJLAB_IMPORT_ERROR
    return list_tasks()


class MjlabVecEnv(VecEnv):
    """Presents an mjlab task as a :class:`VecEnv`.

    Everything about the task comes from its registered configuration: the observation groups, the
    number of actuators, the episode length and the simulation itself. Only the number of
    environments and the device are worth overriding from the outside.

    Unlike the Gymnasium adapter this one exposes no action bounds, because mjlab's actions are
    joint targets rather than a normalized range. Clipping, if a task wants it, is ``clip_actions``,
    applied here before the actions reach the task. A policy exported with
    :meth:`~telekinesis.rlbotics.runner.OnPolicyRunner.export` therefore carries the observation
    normalization but no action scaling, so a deployment has to apply the same clip.

    Example:
        Train a locomotion task with this library's runner::

            env = MjlabVecEnv("Mjlab-Velocity-Flat-Unitree-G1", num_envs=4096, device="cuda")
            runner = OnPolicyRunner(env=env, runner_cfg=cfg, device="cuda")
    """

    def __init__(
        self,
        task: str,
        num_envs: int | None = None,
        device: str = "auto",
        clip_actions: float | None = None,
        play: bool = False,
        render_mode: str | None = None,
        nconmax: int | None = None,
    ) -> None:
        """Create the vector environment.

        Args:
            task: Registered mjlab task id, such as "Mjlab-Velocity-Flat-Unitree-G1". Use
                :func:`registered_tasks` to see what is available.
            num_envs: Number of parallel environments. Defaults to None, which keeps the number the
                task's own scene config asks for.
            device: Device the simulation and the observations live on. Defaults to "auto".
            clip_actions: Symmetric limit applied to the actions before they reach the task.
                Defaults to None, which passes them through.
            play: Whether to load the task's "play" configuration, which is the smaller, less
                randomized variant used for evaluation. Defaults to False.
            render_mode: Passed straight to mjlab. "rgb_array" starts an offscreen renderer so
                :meth:`render` returns frames, which is what recording a video needs. Defaults to
                None, which renders nothing.
            nconmax: Maximum number of contacts mjlab allocates buffers for. Defaults to None,
                which keeps the task's own limit; raise it if a rollout raises "nconmax overflow".

        Raises:
            ImportError: If mjlab is not installed.
            ValueError: If the task id is not registered.
        """
        if ManagerBasedRlEnv is None:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"MjlabVecEnv needs mjlab, which is not installed. {INSTALL_HINT}"
            ) from MJLAB_IMPORT_ERROR

        env_cfg = self._load_cfg(task, play)
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
        if nconmax is not None:
            env_cfg.sim.nconmax = nconmax

        self.task = task
        self.clip_actions = clip_actions
        self.venv = self._build(env_cfg, self._resolve_device(device), render_mode)
        # Read back off the simulation, so the numbers are what mjlab really built
        self.num_envs = self.venv.num_envs
        self.num_actions = self.venv.action_manager.total_action_dim
        self.device = torch.device(self.venv.device)
        self.max_episode_length = int(self.venv.max_episode_length)
        # mjlab does not reset when it is constructed, and the first rollout step needs an
        # observation, so this is where the episode starts
        self.obs = self._observations(self.venv.reset()[0])

    @staticmethod
    def _build(env_cfg, device: str, render_mode: str | None) -> ManagerBasedRlEnv:
        """Build the simulation, growing the contact buffer if the scene needs more than it has.

        mjlab sizes ``sim.nconmax`` for the scene it expects, and a robot that piles up contacts —
        one caught mid-fall on rough terrain, say, or a scene of only a few environments — can exceed
        it, which mjlab reports as an "nconmax must be >= N" :class:`ValueError` at construction.
        Rather than guess a limit upfront, this builds normally and, if that happens, rebuilds once
        with the limit mjlab asked for, doubled. Passing ``nconmax`` explicitly skips the guessing.

        Args:
            env_cfg: The task's environment configuration, already overridden.
            device: Resolved device to simulate on.
            render_mode: Render mode to pass to mjlab.

        Returns:
            The simulation.
        """
        try:
            return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
        except ValueError as error:
            match = re.search(r"nconmax must be >= (\d+)", str(error))
            if match is None:
                raise
            env_cfg.sim.nconmax = int(match.group(1)) * 2
            logger.warning(
                f"   The scene needs more contacts than mjlab allocated, rebuilding with "
                f"nconmax={env_cfg.sim.nconmax}."
            )
            return ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    @staticmethod
    def _load_cfg(task: str, play: bool):
        """Load a task's environment configuration from mjlab's registry.

        Args:
            task: Registered task id.
            play: Whether to load the "play" variant.

        Returns:
            The task's environment configuration.

        Raises:
            ValueError: If the task id is not registered.
        """
        try:
            return load_env_cfg(task, play=play)
        except KeyError as error:
            available = list_tasks()
            examples = ", ".join(available[:5]) if available else "none"
            raise ValueError(
                f"'{task}' is not a registered mjlab task. {len(available)} are registered, "
                f"for example: {examples}. Call "
                "telekinesis.rlbotics.envs.mjlab_env.registered_tasks() for the full list."
            ) from error

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Resolve the device, warning when it is one mjlab cannot train on.

        Args:
            device: Requested device, or "auto".

        Returns:
            A device string mjlab accepts.
        """
        resolved = resolve_device(device)
        if resolved.startswith("mps"):
            logger.warning(
                "   MuJoCo Warp has no 'mps' backend, so mjlab runs on cpu here. Training a task "
                "needs an NVIDIA GPU."
            )
            return "cpu"
        if not resolved.startswith("cuda"):
            logger.warning(
                "   mjlab on cpu is for smoke runs only, since MuJoCo Warp is built around the "
                "GPU. Keep num_envs small."
            )
        return resolved

    @property
    def episode_length_buf(self) -> torch.Tensor:
        """Return mjlab's own per-environment step counter.

        It is delegated rather than duplicated so that writing to it, which is how the runner
        staggers episode lengths at the start of training, reaches the simulation.
        """
        return self.venv.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor) -> None:
        """Write mjlab's per-environment step counter."""
        self.venv.episode_length_buf = value

    def _observations(self, obs: dict[str, torch.Tensor]) -> TensorDict:
        """Wrap mjlab's observation groups in a TensorDict.

        Args:
            obs: Observations, keyed by group name.

        Returns:
            The observations, batched over environments.
        """
        return TensorDict(obs, batch_size=(self.num_envs,))

    @property
    def cfg(self):
        """Return the task's own configuration, read live off the simulation.

        A property rather than a value copied in at construction, so it stays correct even if
        something else mutates the task's config after this adapter is built.
        """
        return self.venv.cfg

    def get_observations(self) -> TensorDict:
        """Return fresh observations without stepping, recomputed rather than replayed.

        Whatever :meth:`step` or :meth:`reset` last cached only reflects the state at that call, so
        a caller in between — logging, or an algorithm peeking before it acts — would otherwise see
        a stale observation if anything else touched the simulation. mjlab's own observation manager
        is the source of truth, the same one :meth:`step` and :meth:`reset` read from.
        """
        return self._observations(self.venv.observation_manager.compute())

    def seed(self, seed: int = -1) -> int:
        """Reseed the task's own random number generator.

        This is separate from :func:`~telekinesis.rlbotics.utils.set_seed`, which seeds model
        init, action sampling and mini-batch order on the training side: the simulation has its
        own domain randomization and reset noise, which lives here instead.

        Args:
            seed: Seed to use. Defaults to -1, which asks mjlab to pick one.

        Returns:
            The seed that was actually used.
        """
        return self.venv.seed(seed)

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
            Observations, rewards and dones of shape (num_envs,), and mjlab's extras. A task with an
            infinite horizon carries the truncation flags under "time_outs" so the algorithm can
            bootstrap episodes cut off by the time limit; a finite-horizon task does not, because
            there the limit is part of the task.
        """
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)

        obs, rewards, terminated, truncated, extras = self.venv.step(actions)
        self.obs = self._observations(obs)

        # mjlab keeps termination and truncation apart, while the contract wants one done flag that
        # can be used arithmetically, plus the truncations for bootstrapping
        if not self.venv.cfg.is_finite_horizon:
            extras["time_outs"] = truncated
        return self.obs, rewards, (terminated | truncated).to(dtype=torch.float32), extras

    def close(self) -> None:
        """Close the underlying environment."""
        self.venv.close()

    def render(self):
        """Return the current frame as an RGB array.

        Only works when the environment was built with ``render_mode="rgb_array"``; otherwise
        mjlab returns None.

        Returns:
            An HxWx3 ``uint8`` array, or None if rendering was not requested.
        """
        return self.venv.render()

    @property
    def render_fps(self) -> float:
        """Return the simulation's step rate, for timing a recorded video."""
        return self.venv.metadata.get("render_fps", 30.0)
