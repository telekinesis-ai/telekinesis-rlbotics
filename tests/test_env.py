"""Tests for vectorized environment modules."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict
from unittest.mock import patch

from telekinesis.rlbotics.envs.base import VecEnv, observation_spec
from telekinesis.rlbotics.envs.gym_env import GymnasiumVecEnv
from telekinesis.rlbotics.envs import isaaclab_env
from telekinesis.rlbotics.envs.isaaclab_env import IsaacLabVecEnv, registered_tasks
from telekinesis.rlbotics.envs.mjlab_env import MjlabVecEnv

# Gymnasium is an optional dependency, so the tests that need a real task are skipped without it
try:
    import gymnasium as gym
except ImportError:  # pragma: no cover - depends on the environment
    gym = None

needs_gymnasium = pytest.mark.skipif(gym is None, reason="Gymnasium is not installed")


class MockVecEnv(VecEnv):
    """Mock vectorized environment for testing.

    VecEnv is an interface with no constructor, so this sets the attributes it declares, which is
    what a real environment does too.
    """

    def __init__(self, num_envs=4, num_actions=3, obs_dim=5, device="cpu"):
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.obs_dim = obs_dim
        self.device = torch.device(device) if isinstance(device, str) else device
        self.max_episode_length = 100
        self.step_count = 0

    def reset(self):
        obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        return TensorDict(
            {"observation": obs},
            batch_size=(self.num_envs,),
            device=self.device,
        )

    def step(self, actions):
        self.step_count += 1
        obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        rewards = torch.randn(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        timeouts = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        obs_dict = TensorDict(
            {"observation": obs},
            batch_size=(self.num_envs,),
            device=self.device,
        )
        extras = {
            "time_outs": timeouts,
            "log": {"/episode/reward": 0.0, "/episode/length": 0.0},
        }

        return obs_dict, rewards, dones, extras

    def get_observations(self):
        return self.reset()


class TestVecEnv:
    """Test VecEnv abstract base class."""

    def test_cannot_instantiate_abstract(self):
        """Test that VecEnv cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VecEnv(num_envs=4, num_actions=3)

    def test_mock_env_initialization(self):
        """Test MockVecEnv initialization."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5)
        assert env.num_envs == 4
        assert env.num_actions == 3
        assert env.device == torch.device("cpu")

    def test_interface_holds_no_state(self):
        """Test that VecEnv stays an interface, with no constructor and nothing to inherit.

        An environment reads its numbers off the simulator it wraps and any episode buffer belongs to
        whoever owns the episode, so a base class allocating either would duplicate or overwrite the
        simulator's own state.
        """
        assert VecEnv.__init__ is object.__init__
        assert not hasattr(VecEnv, "episode_reward_buf")

    def test_reset_returns_tensordict(self):
        """Test that reset returns TensorDict."""
        env = MockVecEnv(num_envs=4, obs_dim=5)
        obs_dict = env.reset()

        assert isinstance(obs_dict, TensorDict)
        assert "observation" in obs_dict
        assert obs_dict["observation"].shape == (4, 5)

    def test_step_returns_correct_types(self):
        """Test that step returns correct types."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5)
        actions = torch.randn(4, 3)

        obs_dict, rewards, dones, extras = env.step(actions)

        assert isinstance(obs_dict, TensorDict)
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(dones, torch.Tensor)
        assert isinstance(extras, dict)

    def test_step_output_shapes(self):
        """Test that step returns correct shapes."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5)
        actions = torch.randn(4, 3)

        obs_dict, rewards, dones, extras = env.step(actions)

        assert obs_dict["observation"].shape == (4, 5)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert "time_outs" in extras
        assert "log" in extras

    def test_observation_spec_is_read_off_the_observations(self):
        """Test that the spec helper derives group shapes without the environment declaring them."""
        env = MockVecEnv(num_envs=4, obs_dim=5)

        assert observation_spec(env) == {"observation": (5,)}

    def test_get_observations(self):
        """Test getting observations without stepping."""
        env = MockVecEnv(num_envs=4, obs_dim=5)
        obs_dict = env.get_observations()

        assert isinstance(obs_dict, TensorDict)
        assert obs_dict["observation"].shape == (4, 5)


class TestVecEnvDevice:
    """Test device handling across environments."""

    def test_cpu_device(self):
        """Test CPU device."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5, device="cpu")
        assert env.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device if available."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5, device="cuda")
        assert env.device == torch.device("cuda")

    def test_device_torch_device(self):
        """Test with torch.device object."""
        device = torch.device("cpu")
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5, device=device)
        assert env.device == device

    def test_observations_on_correct_device(self):
        """Test that observations are on correct device."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=5, device="cpu")
        obs_dict = env.reset()

        assert obs_dict["observation"].device == torch.device("cpu")


class TestVecEnvEdgeCases:
    """Test edge cases and numerical stability."""

    def test_single_environment(self):
        """Test with single environment."""
        env = MockVecEnv(num_envs=1, num_actions=3, obs_dim=5)
        assert env.num_envs == 1

        obs_dict = env.reset()
        assert obs_dict["observation"].shape == (1, 5)

    def test_large_batch_size(self):
        """Test with large batch size."""
        env = MockVecEnv(num_envs=256, num_actions=3, obs_dim=5)
        obs_dict = env.reset()
        assert obs_dict["observation"].shape == (256, 5)

    def test_high_dimensional_observations(self):
        """Test with high-dimensional observations."""
        env = MockVecEnv(num_envs=4, num_actions=3, obs_dim=1024)
        obs_dict = env.reset()
        assert obs_dict["observation"].shape == (4, 1024)

    def test_many_actions(self):
        """Test with many actions."""
        env = MockVecEnv(num_envs=4, num_actions=128, obs_dim=5)
        actions = torch.randn(4, 128)
        obs_dict, rewards, dones, extras = env.step(actions)
        assert obs_dict["observation"].shape == (4, 5)


@needs_gymnasium
class TestGymnasiumVecEnv:
    """Test the Gymnasium adapter, which should work from an environment id alone."""

    def test_reads_pendulum_from_the_id(self):
        """Test that the observation size, actions, bounds and episode limit come from the spec."""
        env = GymnasiumVecEnv("Pendulum-v1", num_envs=2, device="cpu")

        assert env.obs_dim == 3
        assert env.num_actions == 1
        assert env.max_episode_length == 200
        assert torch.allclose(env.action_low, torch.tensor([-2.0]))
        assert torch.allclose(env.action_high, torch.tensor([2.0]))
        env.close()

    def test_reads_a_different_task_from_the_id(self):
        """Test that switching the id is all a different continuous task needs."""
        env = GymnasiumVecEnv("MountainCarContinuous-v0", num_envs=2, device="cpu")

        assert env.obs_dim == 2
        assert env.num_actions == 1
        assert observation_spec(env) == {"observation": (2,)}
        env.close()

    def test_step_shapes_and_bounds(self):
        """Test that a step returns the documented shapes and scales actions onto the bounds."""
        env = GymnasiumVecEnv("Pendulum-v1", num_envs=4, device="cpu")

        # Beyond [-1, 1], so the clipping and the scaling both have to act
        obs, rewards, dones, extras = env.step(torch.full((4, 1), 5.0))

        assert obs["observation"].shape == (4, 3)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert extras["time_outs"].shape == (4,)
        assert extras["time_outs"].dtype == torch.bool
        env.close()

    def test_discrete_action_space_is_rejected(self):
        """Test that a discrete task fails immediately, naming what is supported."""
        with pytest.raises(ValueError, match="only continuous"):
            GymnasiumVecEnv("CartPole-v1", num_envs=2, device="cpu")

    def test_non_flat_observation_is_rejected(self):
        """Test that an image-like observation fails instead of being silently flattened."""

        class ImageEnv(gym.Env):
            observation_space = gym.spaces.Box(0, 255, (3, 8, 8), dtype=np.uint8)
            action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                return self.observation_space.sample(), {}

            def step(self, action):
                return self.observation_space.sample(), 0.0, False, False, {}

        gym.register("TestImage-v0", entry_point=lambda **kwargs: ImageEnv())
        with pytest.raises(ValueError, match="only flat Box observations"):
            GymnasiumVecEnv("TestImage-v0", num_envs=2, device="cpu")

    def test_unbounded_actions_are_passed_through(self):
        """Test that a task without action bounds trains without emitting infinite actions."""

        class UnboundedEnv(gym.Env):
            observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32)
            action_space = gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                return np.zeros(2, dtype=np.float32), {}

            def step(self, action):
                return np.asarray(action.repeat(2), dtype=np.float32), 0.0, False, False, {}

        gym.register("TestUnbounded-v0", entry_point=lambda **kwargs: UnboundedEnv())
        env = GymnasiumVecEnv("TestUnbounded-v0", num_envs=2, device="cpu")

        assert env.action_low is None
        assert env.action_high is None

        obs, _, _, _ = env.step(torch.full((2, 1), 3.0))
        assert torch.isfinite(obs["observation"]).all()
        # Passed through, not squashed into [-1, 1]
        assert torch.allclose(obs["observation"], torch.full((2, 2), 3.0))
        env.close()

    def test_missing_gymnasium_points_at_the_extra(self):
        """Test that the import error tells the user what to install."""
        with patch("telekinesis.rlbotics.envs.gym_env.gym", None):
            with pytest.raises(ImportError, match=r"telekinesis-rlbotics\[gym\]"):
                GymnasiumVecEnv("Pendulum-v1", num_envs=2, device="cpu")


class _FakeEnvCfg:
    """Stand-in for a manager-based env config, which carries num_envs on its scene."""

    def __init__(
        self,
        num_envs: int = 4096,
        groups: dict[str, int] | None = None,
        is_finite_horizon: bool = False,
    ):
        self.scene = SimpleNamespace(num_envs=num_envs)
        self.sim = SimpleNamespace(nconmax=None)
        self.groups = groups or {"actor": 48, "critic": 60}
        self.is_finite_horizon = is_finite_horizon


class _FakeSimEnv:
    """Stand-in for a manager-based env, with the surface both adapters read.

    Both adapters drive this directly, so it mirrors ManagerBasedRlEnv: observations as a plain dict,
    termination and truncation reported separately, and no reset on construction. ``unwrapped``
    stands in for the Gymnasium wrapper chain that ``gym.make`` puts around an Isaac Lab task.
    """

    def __init__(self, cfg, device, render_mode=None):
        self.cfg = cfg
        self.device = device
        self.render_mode = render_mode
        self.num_envs = cfg.scene.num_envs
        self.action_manager = SimpleNamespace(total_action_dim=12)
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long)
        self.metadata = {"render_fps": 50.0}
        self.closed = False
        self.reset_count = 0
        self.stepped_with = None

    @property
    def unwrapped(self):
        return self

    def _obs(self) -> dict[str, torch.Tensor]:
        return {
            group: torch.randn(self.num_envs, dim) for group, dim in self.cfg.groups.items()
        }

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self._obs()

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict]:
        self.reset_count += 1
        return self._obs(), {"log": {}}

    def step(self, action):
        self.stepped_with = action
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated[0] = True
        return self._obs(), torch.randn(self.num_envs), terminated, truncated, {"log": {}}

    def render(self):
        return None if self.render_mode is None else torch.zeros(4, 4, 3).numpy()

    def close(self) -> None:
        self.closed = True


def fake_mjlab(tasks: tuple[str, ...] = ("Mjlab-Velocity-Flat-Unitree-G1",), **cfg_kwargs):
    """Patch the mjlab symbols the adapter imports with the stubs above.

    Args:
        tasks: Task ids the stubbed registry knows about.
        **cfg_kwargs: Passed to the stubbed environment config.

    Returns:
        A patch context manager.
    """

    def load_env_cfg(task_name, play=False):
        if task_name not in tasks:
            raise KeyError(task_name)
        return _FakeEnvCfg(**cfg_kwargs)

    return patch.multiple(
        "telekinesis.rlbotics.envs.mjlab_env",
        ManagerBasedRlEnv=_FakeSimEnv,
        load_env_cfg=load_env_cfg,
        list_tasks=lambda: list(tasks),
    )


class TestMjlabVecEnv:
    """Test the mjlab adapter against a stubbed mjlab, since the real one needs an NVIDIA GPU."""

    task = "Mjlab-Velocity-Flat-Unitree-G1"

    def test_reads_the_task_configuration(self):
        """Test that the numbers come from what mjlab built, not from the caller."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=64, device="cpu")

            assert env.num_envs == 64
            assert env.num_actions == 12
            assert env.max_episode_length == 1000
            assert env.device == torch.device("cpu")

    def test_num_envs_defaults_to_the_task(self):
        """Test that the task's own scene config decides when nothing is passed."""
        with fake_mjlab(num_envs=4096):
            env = MjlabVecEnv(self.task, device="cpu")

            assert env.num_envs == 4096

    def test_observation_spec_covers_every_group(self):
        """Test that the spec is derived from the task's observation groups."""
        with fake_mjlab(groups={"actor": 48, "critic": 60}):
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")

            assert observation_spec(env) == {"actor": (48,), "critic": (60,)}

    def test_step_returns_the_documented_shapes(self):
        """Test that dones come back as float flags and truncations are passed through."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")
            obs, rewards, dones, extras = env.step(torch.zeros(8, 12))

            assert obs["actor"].shape == (8, 48)
            assert rewards.shape == (8,)
            assert dones.shape == (8,)
            assert dones.dtype == torch.float32
            assert extras["time_outs"].shape == (8,)

    def test_finite_horizon_task_gets_no_time_outs(self):
        """Test that a task whose time limit is part of the task is not bootstrapped.

        The limit is the end of the episode there, not an artificial cutoff, so reporting it as a
        truncation would have the algorithm add a value estimate for a state that has no future.
        """
        with fake_mjlab(is_finite_horizon=True):
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")
            _, _, dones, extras = env.step(torch.zeros(8, 12))

            assert "time_outs" not in extras
            # The truncation still ends the episode, it is just not bootstrapped
            assert dones[0] == 1.0

    def test_actions_are_clipped_before_the_simulation(self):
        """Test that clip_actions limits what reaches the task, not just what is recorded."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu", clip_actions=0.5)
            env.step(torch.full((8, 12), 5.0))

            assert env.venv.stepped_with.max() == 0.5
            assert env.venv.stepped_with.min() == 0.5

    def test_actions_are_passed_through_without_a_clip(self):
        """Test that no clip means the actions reach the task untouched."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")
            env.step(torch.full((8, 12), 5.0))

            assert env.venv.stepped_with.max() == 5.0

    def test_construction_leaves_the_simulator_counter_alone(self):
        """Test that building the adapter does not replace mjlab's own episode counter.

        The counter belongs to the simulation, whose termination terms read it every step, so the
        adapter reads and writes through to it instead of allocating one of its own.
        """
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")
            sim_buffer = env.venv.episode_length_buf

            assert env.episode_length_buf is sim_buffer

    def test_resets_on_construction(self):
        """Test that the task is reset when it is built, since mjlab does not do it itself."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")

            assert env.venv.reset_count == 1
            assert env.get_observations()["actor"].shape == (8, 48)

    def test_episode_length_buf_is_delegated(self):
        """Test that staggering episode lengths reaches the simulation's own counter."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")
            env.episode_length_buf = torch.full((8,), 7, dtype=torch.long)

            assert torch.equal(env.venv.episode_length_buf, torch.full((8,), 7, dtype=torch.long))
            assert torch.equal(env.episode_length_buf, env.venv.episode_length_buf)

    def test_unknown_task_lists_what_is_registered(self):
        """Test that a wrong task id says so and points at the registry."""
        with fake_mjlab(tasks=("Mjlab-Velocity-Flat-Unitree-G1", "Mjlab-Tracking-Flat-Unitree-G1")):
            with pytest.raises(ValueError, match="not a registered mjlab task"):
                MjlabVecEnv("Mjlab-Nope", device="cpu")

    def test_mps_falls_back_to_cpu(self):
        """Test that Apple Silicon does not reach MuJoCo Warp, which has no Metal backend."""
        with fake_mjlab():
            with patch(
                "telekinesis.rlbotics.envs.mjlab_env.resolve_device", return_value="mps"
            ):
                env = MjlabVecEnv(self.task, num_envs=4, device="auto")

            assert env.venv.device == "cpu"

    def test_reset_returns_observations_only(self):
        """Test that mjlab's (observations, extras) reset is adapted to the VecEnv contract."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=8, device="cpu")
            obs = env.reset()

            assert isinstance(obs, TensorDict)
            assert obs["actor"].shape == (8, 48)

    def test_close_closes_mjlab(self):
        """Test that closing reaches the simulation."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=4, device="cpu")
            env.close()

            assert env.venv.closed

    def test_render_needs_a_render_mode(self):
        """Test that frames come back only when the task was built to render."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=4, device="cpu")
            assert env.render() is None
            assert env.render_fps == 50.0

            recording = MjlabVecEnv(self.task, num_envs=4, device="cpu", render_mode="rgb_array")
            assert recording.render() is not None

    def test_missing_mjlab_points_at_the_extra(self):
        """Test that the import error tells the user what to install."""
        with patch("telekinesis.rlbotics.envs.mjlab_env.ManagerBasedRlEnv", None):
            with pytest.raises(ImportError, match=r"telekinesis-rlbotics\[mjlab\]"):
                MjlabVecEnv(self.task, device="cpu")


def fake_isaaclab(
    tasks: tuple[str, ...] = ("Isaac-Velocity-Flat-Anymal-C-v0",),
    groups: dict[str, int] | None = None,
    num_envs: int = 4096,
    is_finite_horizon: bool = False,
):
    """Patch the Isaac Lab pieces the adapter imports with stubs.

    Isaac Lab imports its task modules only once Isaac Sim is running, so the adapter defers them to
    ``_load_isaaclab`` and launches the app first. Both are patched here. The adapter drives the task
    ``gym.make`` returns directly, so the stubbed registry hands back a ``_FakeSimEnv``.

    Args:
        tasks: Task ids the stubbed Gymnasium registry knows about.
        groups: Observation groups and their sizes.
        num_envs: Environment count the task config falls back to.
        is_finite_horizon: Whether the task's horizon is finite, which decides whether the adapter
            publishes truncations for bootstrapping.

    Returns:
        A patch context manager.
    """

    class Error(Exception):
        """Stand-in for gymnasium.error.Error."""

    def make(task, cfg=None, render_mode=None):
        if task not in tasks:
            raise Error(f"unknown task {task}")
        return _FakeSimEnv(cfg, cfg.device)

    # Includes a non-Isaac id, so the "Isaac-" filter has something to exclude
    gym = SimpleNamespace(
        registry=dict.fromkeys([*tasks, "CartPole-v1"]),
        make=make,
        error=SimpleNamespace(Error=Error),
    )

    def parse_env_cfg(task_name, device="cuda:0", num_envs=None, use_fabric=None):
        cfg = _FakeEnvCfg(
            num_envs=num_envs or 4096,
            groups=groups or {"policy": 48, "critic": 60},
            is_finite_horizon=is_finite_horizon,
        )
        cfg.device = device
        return cfg

    return patch.multiple(
        "telekinesis.rlbotics.envs.isaaclab_env",
        launch_simulator=lambda **kwargs: SimpleNamespace(close=lambda: None),
        _load_isaaclab=lambda: (gym, parse_env_cfg),
    )


class TestIsaacLabVecEnv:
    """Test the Isaac Lab adapter against stubs, since Isaac Sim needs Linux and an NVIDIA GPU."""

    task = "Isaac-Velocity-Flat-Anymal-C-v0"

    def test_reads_the_task_configuration(self):
        """Test that the numbers come from what Isaac Lab built, not from the caller."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=64, device="cpu")

            assert env.num_envs == 64
            assert env.num_actions == 12
            assert env.max_episode_length == 1000
            assert env.device == torch.device("cpu")

    def test_observation_spec_covers_every_group(self):
        """Test that the spec is derived from the task's observation groups."""
        with fake_isaaclab(groups={"policy": 48, "critic": 60}):
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu")

            assert observation_spec(env) == {"policy": (48,), "critic": (60,)}

    def test_step_returns_the_documented_shapes(self):
        """Test that dones come back as float flags and truncations are passed through."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu")
            obs, rewards, dones, extras = env.step(torch.zeros(8, 12))

            assert obs["policy"].shape == (8, 48)
            assert rewards.shape == (8,)
            assert dones.shape == (8,)
            assert dones.dtype == torch.float32
            assert extras["time_outs"].shape == (8,)

    def test_actions_are_clipped_before_the_simulation(self):
        """Test that clip_actions limits what reaches the task, not just what is recorded."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu", clip_actions=0.5)
            env.step(torch.full((8, 12), 5.0))

            assert env.venv.stepped_with.max() == 0.5
            assert env.venv.stepped_with.min() == 0.5

    def test_actions_are_passed_through_without_a_clip(self):
        """Test that no clip means the actions reach the task untouched."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu")
            env.step(torch.full((8, 12), 5.0))

            assert env.venv.stepped_with.max() == 5.0

    def test_a_finite_horizon_task_does_not_bootstrap_time_outs(self):
        """Test that "time_outs" is only published when the task's horizon is infinite."""
        with fake_isaaclab(is_finite_horizon=True):
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu")
            _, _, dones, extras = env.step(torch.zeros(8, 12))

            assert "time_outs" not in extras
            # The truncation still ends the episode, it is just not bootstrapped
            assert dones[0] == 1.0

    def test_episode_length_buf_is_delegated(self):
        """Test that staggering episode lengths reaches the simulation's own counter."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu")
            env.episode_length_buf = torch.full((8,), 7, dtype=torch.long)

            assert torch.equal(env.venv.episode_length_buf, torch.full((8,), 7, dtype=torch.long))

    def test_unknown_task_lists_what_is_registered(self):
        """Test that a wrong task id says so and reports only Isaac Lab's tasks."""
        with fake_isaaclab():
            with pytest.raises(ValueError, match="not a registered Isaac Lab task") as error:
                IsaacLabVecEnv("Isaac-Nope-v0", device="cpu")

            assert "1 are registered" in str(error.value)
            assert "CartPole-v1" not in str(error.value)

    def test_registered_tasks_filters_to_isaac_lab(self):
        """Test that the task list is Gymnasium's registry narrowed to Isaac Lab's entries."""
        with fake_isaaclab(tasks=("Isaac-Cartpole-v0", "Isaac-Ant-v0")):
            assert registered_tasks() == ["Isaac-Ant-v0", "Isaac-Cartpole-v0"]

    def test_mps_falls_back_to_cpu(self):
        """Test that Apple Silicon does not reach Isaac Sim, which has no macOS build."""
        with fake_isaaclab():
            with patch(
                "telekinesis.rlbotics.envs.isaaclab_env.resolve_device", return_value="mps"
            ):
                env = IsaacLabVecEnv(self.task, num_envs=4, device="auto")

            assert env.venv.device == "cpu"

    def test_reset_returns_observations_only(self):
        """Test that Isaac Lab's (observations, extras) reset is adapted to the VecEnv contract."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=8, device="cpu")
            obs = env.reset()

            assert isinstance(obs, TensorDict)
            assert obs["policy"].shape == (8, 48)

    def test_close_leaves_the_simulator_running(self):
        """Test that closing the task leaves Isaac Sim running, as a process-wide singleton."""
        with fake_isaaclab():
            env = IsaacLabVecEnv(self.task, num_envs=4, device="cpu")
            env.close()

            assert env.venv.closed
            assert isaaclab_env._SIMULATION_APP is None

    def test_missing_isaaclab_points_at_the_extra(self):
        """Test that the import error tells the user what to install, index URL included."""
        with patch.object(isaaclab_env, "ISAACLAB_AVAILABLE", False):
            with pytest.raises(ImportError, match="pypi.nvidia.com"):
                isaaclab_env.launch_simulator()

    def test_shutdown_closes_the_simulator(self):
        """Test that shutting down closes the app and forgets it, so a later launch starts fresh."""
        closed = []
        app = SimpleNamespace(close=lambda: closed.append(True))
        with patch.object(isaaclab_env, "_SIMULATION_APP", app):
            isaaclab_env.shutdown_simulator()

            assert closed == [True]
            assert isaaclab_env._SIMULATION_APP is None


class TestMjlabContactBuffer:
    """Test growing mjlab's contact buffer, which a small or falling scene can overflow."""

    task = "Mjlab-Velocity-Flat-Unitree-G1"

    def _flaky_env(self, message: str, failures: int = 1):
        """Return a ManagerBasedRlEnv stand-in that raises the given message the first N times."""
        state = {"attempts": 0}

        class Flaky(_FakeSimEnv):
            def __init__(self, cfg, device, render_mode=None):
                state["attempts"] += 1
                if state["attempts"] <= failures:
                    raise ValueError(message)
                super().__init__(cfg, device, render_mode)

        return Flaky, state

    def test_an_overflow_is_retried_with_the_limit_mjlab_asked_for(self):
        """A contact-buffer overflow rebuilds once with double the reported minimum."""
        Flaky, state = self._flaky_env("nconmax must be >= 123, got 64")
        with fake_mjlab():
            with patch("telekinesis.rlbotics.envs.mjlab_env.ManagerBasedRlEnv", Flaky):
                env = MjlabVecEnv(self.task, num_envs=1, device="cpu")

            assert state["attempts"] == 2
            assert env.venv.cfg.sim.nconmax == 246

    def test_an_explicit_limit_is_kept(self):
        """Passing nconmax skips the guessing, and is what reaches mjlab."""
        with fake_mjlab():
            env = MjlabVecEnv(self.task, num_envs=1, device="cpu", nconmax=500)

            assert env.venv.cfg.sim.nconmax == 500

    def test_other_errors_are_not_swallowed(self):
        """A failure that is not a contact overflow propagates, rather than being retried."""
        Flaky, state = self._flaky_env("something else went wrong")
        with fake_mjlab():
            with patch("telekinesis.rlbotics.envs.mjlab_env.ManagerBasedRlEnv", Flaky):
                with pytest.raises(ValueError, match="something else"):
                    MjlabVecEnv(self.task, num_envs=1, device="cpu")

            assert state["attempts"] == 1

    def test_a_second_overflow_is_not_retried_again(self):
        """One rebuild is the limit, so a scene that keeps overflowing fails instead of looping."""
        Flaky, state = self._flaky_env("nconmax must be >= 123", failures=2)
        with fake_mjlab():
            with patch("telekinesis.rlbotics.envs.mjlab_env.ManagerBasedRlEnv", Flaky):
                with pytest.raises(ValueError, match="nconmax"):
                    MjlabVecEnv(self.task, num_envs=1, device="cpu")

            assert state["attempts"] == 2
