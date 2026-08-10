"""Tests for the on-policy runner."""

import importlib.util
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from unittest.mock import patch
from tensordict import TensorDict

from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
    SymmetryConfig,
)
from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.algorithms import PPO
from telekinesis.rlbotics.runner import OnPolicyRunner

# Check if onnx is available
HAS_ONNX = importlib.util.find_spec("onnx") is not None


class MockEnvironment(VecEnv):
    """Simple mock environment for testing, optionally exposing a privileged observation group."""

    def __init__(
        self,
        num_envs: int = 4,
        obs_dim: int = 8,
        num_actions: int = 2,
        device: str = "cpu",
        priv_dim: int = 0,
    ):
        """Initialize the mock environment."""
        # VecEnv is an interface, so the environment sets the attributes it promises
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.priv_dim = priv_dim
        self.max_episode_length = 100
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.cfg = {"obs_dim": obs_dim, "num_actions": num_actions}
        self.obs = torch.randn(num_envs, obs_dim, device=device)
        self.priv = torch.randn(num_envs, priv_dim, device=device) if priv_dim else None
        self.step_count = 0

    def _observations(self) -> TensorDict:
        """Build the observation TensorDict."""
        obs = {"observation": self.obs}
        if self.priv_dim:
            obs["privileged"] = self.priv
        return TensorDict(obs, batch_size=(self.num_envs,), device=self.device)

    def reset(self) -> TensorDict:
        """Reset all environments."""
        self.obs = torch.randn(self.num_envs, self.obs_dim, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        return self._observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments."""
        self.obs = self.obs + 0.01 * torch.randn_like(self.obs)
        rewards = torch.randn(self.num_envs, device=self.device)
        dones = (torch.rand(self.num_envs, device=self.device) < 0.05).float()

        self.step_count += 1
        timeouts = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        if self.step_count >= self.max_episode_length:
            timeouts = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
            dones = torch.ones(self.num_envs, device=self.device)
            self.step_count = 0

        extras = {
            "time_outs": timeouts,
            "log": {"episode_reward": rewards.mean().item()},
        }

        return self._observations(), rewards, dones, extras

    def get_observations(self) -> TensorDict:
        """Return current observations."""
        return self._observations()


def make_runner_cfg(
    num_steps_per_env: int = 1,
    save_interval: int = 100,
    log_dir: str | None = None,
    experiment: str = "test_run",
    critic_groups: list[str] | None = None,
    resume: str | None = None,
    log_video: bool = False,
    **overrides,
) -> OnPolicyRunnerConfig:
    """Build a small runner config for the mock environment.

    Args:
        num_steps_per_env: Steps collected per environment per iteration.
        save_interval: Iterations between checkpoints.
        log_dir: Root directory for the run. None writes nothing.
        experiment: Experiment name, the directory grouping runs under log_dir.
        critic_groups: Observation groups the critic consumes. Defaults to the actor's groups.
        resume: Which checkpoint to continue from: "last", "best", a name or a path. None trains
            from scratch.
        log_video: Whether the runner records a clip beside each checkpoint.
        **overrides: Additional runner config fields.

    Returns:
        The runner configuration.
    """
    return OnPolicyRunnerConfig(
        obs_groups={"actor": ["observation"], "critic": critic_groups or ["observation"]},
        num_steps_per_env=num_steps_per_env,
        logger=LoggerConfig(
            log_dir=log_dir,
            experiment=experiment,
            log_interval=1,
            save_interval=save_interval,
            resume=resume,
            log_video=log_video,
        ),
        algorithm=PPOConfig(num_learning_epochs=1, num_mini_batches=1),
        actor=MLPConfig(hidden_dims=(16, 16), distribution_cfg=GaussianDistributionConfig()),
        critic=MLPConfig(hidden_dims=(16, 16)),
        **overrides,
    )


class TestOnPolicyRunnerInitialization:
    """Test OnPolicyRunner initialization."""

    def test_initialization(self):
        """Test basic runner initialization."""
        env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")
        runner_cfg = make_runner_cfg(num_steps_per_env=5)

        runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device="cpu")

        assert runner.env == env
        assert isinstance(runner.alg, PPO)
        assert runner.runner_cfg == runner_cfg
        assert runner.device == "cpu"
        assert runner.current_learning_iteration == 0
        assert runner.logger is not None

    def test_models_sized_from_environment(self):
        """Test that the models are sized from the observations and the action space."""
        env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")

        runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")

        assert runner.alg.actor.input_dim == 8
        assert runner.alg.actor.output_dim == 2
        assert runner.alg.critic.input_dim == 8
        assert runner.alg.critic.output_dim == 1

    def test_privileged_critic_observations(self):
        """Test that the critic can consume more observation groups than the actor."""
        env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu", priv_dim=5)
        runner_cfg = make_runner_cfg(critic_groups=["observation", "privileged"])

        runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device="cpu")

        assert runner.alg.actor.input_dim == 8
        assert runner.alg.critic.input_dim == 13
        # The critic observes a different set, so it gets its own rollout buffer
        assert runner.alg.storage.has_critic_obs
        assert runner.alg.storage.critic_obs_shape == (13,)

    def test_initialization_with_log_dir(self):
        """Test runner initialization with log directory."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(
                env=env, runner_cfg=make_runner_cfg(num_steps_per_env=5, log_dir=log_dir), device="cpu"
            )

            assert runner.logger.log_dir.startswith(log_dir)

    def test_run_directory_is_experiment_then_timestamp(self):
        """Test the run lands in <log_dir>/<experiment>/<timestamp>."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(log_dir=log_dir, experiment="my_experiment"),
                device="cpu",
            )

            run_dir = Path(runner.log_dir)
            assert run_dir.parent == Path(log_dir) / "my_experiment"
            assert run_dir.parent.parent == Path(log_dir)

    def test_config_built_from_dict(self):
        """Test that a config loaded from a dictionary, as a YAML file would be, drives the runner."""
        env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")
        runner_cfg = OnPolicyRunnerConfig.from_dict({
            "obs_groups": {"actor": ["observation"], "critic": ["observation"]},
            "num_steps_per_env": 2,
            "actor": {"hidden_dims": [16], "distribution_cfg": {"init_std": 0.5}},
            "critic": {"hidden_dims": [16]},
        })

        runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device="cpu")

        assert runner.runner_cfg.num_steps_per_env == 2
        assert runner.alg.actor.mlp[0].out_features == 16

    def test_unknown_observation_group_is_rejected(self):
        """Test that an unknown observation group fails before training starts."""
        env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")

        with pytest.raises(ValueError, match="not found in the observations"):
            OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(critic_groups=["does_not_exist"]),
                device="cpu",
            )


class TestInitAtRandomEpisodeLength:
    """Test staggering the initial episode lengths, which needs the environment's own counter."""

    def test_randomizes_when_the_environment_tracks_episodes(self):
        """Test that the counter is filled with lengths below the episode limit."""
        env = MockEnvironment(num_envs=8)
        runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")

        runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)

        assert env.episode_length_buf.shape == (8,)
        assert env.episode_length_buf.max() < env.max_episode_length

    def test_warns_when_the_environment_has_no_counter(self):
        """Test that an environment without a counter is told, rather than silently ignored.

        episode_length_buf is optional on the interface, so the request cannot be honoured here; the
        run continues from step 0 either way, and saying so beats pretending it worked.
        """
        env = MockEnvironment(num_envs=8)
        del env.episode_length_buf
        runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")

        with patch("telekinesis.rlbotics.runner.logger") as mock_logger:
            runner.learn(num_learning_iterations=1, init_at_random_ep_len=True)

        assert mock_logger.warning.called
        assert "episode_length_buf" in mock_logger.warning.call_args[0][0]


class TestOnPolicyRunnerLearn:
    """Test the learning loop."""

    def test_learn_basic(self):
        """Test basic learning loop execution."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(
                    num_steps_per_env=2, save_interval=10, log_dir=log_dir
                ),
                device="cpu",
            )
            before = runner.alg.actor.mlp[0].weight.clone()

            runner.learn(num_learning_iterations=2)

            # The policy was optimized and the storage was consumed
            assert not torch.allclose(before, runner.alg.actor.mlp[0].weight)
            assert runner.alg.storage.step == 0
            assert runner.current_learning_iteration >= 1

    def test_learn_iterations(self):
        """Test that correct number of iterations are executed."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(
                    num_steps_per_env=1, save_interval=10, log_dir=log_dir
                ),
                device="cpu",
            )

            runner.learn(num_learning_iterations=5)

            assert runner.current_learning_iteration == 4

    def test_learn_records_metrics(self):
        """Test that the rollout is counted and the action std is reported."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(
                    num_steps_per_env=2, save_interval=10, log_dir=log_dir
                ),
                device="cpu",
            )

            runner.learn(num_learning_iterations=2)

            assert runner.logger.total_timesteps == 2 * 2 * 2
            assert runner.alg.get_action_std() is not None

    def test_learn_rejects_nan_observations(self):
        """Test that NaN from the environment is caught during the rollout."""
        env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")
        env.obs = torch.full_like(env.obs, float("nan"))

        runner = OnPolicyRunner(
            env=env,
            runner_cfg=make_runner_cfg(num_steps_per_env=2, check_for_nan=True),
            device="cpu",
        )

        with pytest.raises(ValueError, match="NaN"):
            runner.learn(num_learning_iterations=1)


class TestOnPolicyRunnerSaveLoad:
    """Test save and load functionality."""

    def test_save_creates_file(self):
        """Test that save creates a checkpoint file."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")
            runner.current_learning_iteration = 10

            checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
            runner.save(checkpoint_path)

            assert os.path.exists(checkpoint_path)

    def test_save_contains_correct_data(self):
        """Test that saved checkpoint contains correct data."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")

            runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")
            runner.current_learning_iteration = 42

            checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
            runner.save(checkpoint_path)

            loaded = torch.load(checkpoint_path, weights_only=False)
            assert loaded["iteration"] == 42
            assert "actor_state_dict" in loaded
            assert "critic_state_dict" in loaded
            assert "optimizer_state_dict" in loaded

    def test_load_restores_state(self):
        """Test that load restores runner state."""
        with tempfile.TemporaryDirectory() as log_dir:
            # Save
            env1 = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")
            runner_cfg = make_runner_cfg()
            runner1 = OnPolicyRunner(env=env1, runner_cfg=runner_cfg, device="cpu")
            runner1.current_learning_iteration = 25
            checkpoint_path = os.path.join(log_dir, "checkpoint.pt")
            runner1.save(checkpoint_path)

            # Load in new runner
            env2 = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")
            runner2 = OnPolicyRunner(env=env2, runner_cfg=runner_cfg, device="cpu")

            runner2.load(checkpoint_path)

            assert runner2.current_learning_iteration == 25
            # The loaded weights match the saved ones
            assert torch.allclose(runner1.alg.actor.mlp[0].weight, runner2.alg.actor.mlp[0].weight)


class TestOnPolicyRunnerInference:
    """Test inference-related functionality."""

    def test_get_inference_policy(self):
        """Test getting inference policy."""
        env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")

        runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")

        policy = runner.get_inference_policy()

        assert policy is not None
        # The algorithm was switched to evaluation mode
        assert runner.alg.actor.training is False

    def test_inference_policy_produces_actions(self):
        """Test that the inference policy maps observations to actions."""
        env = MockEnvironment(num_envs=2, obs_dim=8, num_actions=2, device="cpu")

        runner = OnPolicyRunner(env=env, runner_cfg=make_runner_cfg(), device="cpu")
        policy = runner.get_inference_policy()

        with torch.no_grad():
            actions = policy(env.get_observations()["observation"], stochastic=False)

        assert actions.shape == (2, 2)


@pytest.mark.skipif(not HAS_ONNX, reason="onnx is not installed")
class TestExport:
    """Test exporting the trained policy for deployment."""

    def _runner(self, env: MockEnvironment | None = None) -> OnPolicyRunner:
        """Build a runner on the mock environment."""
        return OnPolicyRunner(
            env=env or MockEnvironment(num_envs=4, obs_dim=8, num_actions=2),
            runner_cfg=make_runner_cfg(),
            device="cpu",
        )

    def test_writes_a_single_file(self):
        """Test that the weights stay in the graph instead of going to a sidecar file."""
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = runner.export(path=temp_dir)

            assert path.name == "policy.onnx"
            assert [entry.name for entry in Path(temp_dir).iterdir()] == ["policy.onnx"]

    def test_is_self_contained(self):
        """Test that the exported file loads on its own, moved away from where it was written."""
        onnxruntime = pytest.importorskip("onnxruntime")
        runner = self._runner()
        with tempfile.TemporaryDirectory() as export_dir, tempfile.TemporaryDirectory() as elsewhere:
            path = runner.export(path=export_dir)
            moved = Path(elsewhere) / "moved.onnx"
            moved.write_bytes(path.read_bytes())

            session = onnxruntime.InferenceSession(str(moved))
            assert [i.name for i in session.get_inputs()] == ["observation"]
            assert [o.name for o in session.get_outputs()] == ["action"]

    def test_matches_the_trained_policy(self):
        """Test that the exported graph computes what the torch policy computes."""
        from telekinesis.rlbotics.policy import Policy

        pytest.importorskip("onnxruntime")
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = Policy(runner.export(path=temp_dir))

            obs = torch.randn(5, 8)
            with torch.no_grad():
                expected = runner.alg.get_policy().to("cpu").eval().as_onnx().eval()(obs)

            assert torch.allclose(
                torch.from_numpy(policy.get_action(obs.numpy())), expected, atol=1e-5
            )

    def test_batch_dimension_is_dynamic(self):
        """Test that a deployment can act on one observation or on many."""
        from telekinesis.rlbotics.policy import Policy

        pytest.importorskip("onnxruntime")
        runner = self._runner()
        with tempfile.TemporaryDirectory() as temp_dir:
            policy = Policy(runner.export(path=temp_dir))

            assert policy.get_action(torch.randn(8).numpy()).shape == (2,)
            assert policy.get_action(torch.randn(1, 8).numpy()).shape == (1, 2)
            assert policy.get_action(torch.randn(7, 8).numpy()).shape == (7, 2)

    def test_action_bounds_are_baked_in(self):
        """Test that an environment with action bounds gets them folded into the graph."""
        from telekinesis.rlbotics.policy import Policy

        pytest.importorskip("onnxruntime")
        env = MockEnvironment(num_envs=4, obs_dim=8, num_actions=2)
        env.action_low = torch.tensor([-0.5, -3.0])
        env.action_high = torch.tensor([0.5, 3.0])
        runner = self._runner(env)

        with tempfile.TemporaryDirectory() as temp_dir:
            policy = Policy(runner.export(path=temp_dir))

            # Large observations drive the pre-scaling action outside [-1, 1]
            actions = policy.get_action((torch.randn(64, 8) * 100).numpy())
            assert (actions[:, 0] >= -0.5).all() and (actions[:, 0] <= 0.5).all()
            assert (actions[:, 1] >= -3.0).all() and (actions[:, 1] <= 3.0).all()

    def test_defaults_to_the_run_directory(self):
        """Test that the export lands next to the run's checkpoints when no path is given."""
        with tempfile.TemporaryDirectory() as temp_dir:
            runner = OnPolicyRunner(
                env=MockEnvironment(num_envs=4, obs_dim=8, num_actions=2),
                runner_cfg=make_runner_cfg(log_dir=temp_dir),
                device="cpu",
            )
            path = runner.export()

            assert path.parent == Path(runner.log_dir)

    def test_without_a_directory_raises(self):
        """Test that exporting with logging off and no path is an error, not a silent no-op."""
        runner = self._runner()
        with pytest.raises(ValueError, match="nowhere to export to"):
            runner.export()


class TestBestCheckpoint:
    """Test that training keeps the best policy and that resuming can start from it."""

    def test_training_writes_the_best_checkpoint(self):
        """A run keeps model_best.pt, scored on the mean episode reward."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=4)
            # Episodes end every other step, so some have finished by the time it is scored
            env.max_episode_length = 2
            runner = OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(log_dir=log_dir, save_interval=100),
                device="cpu",
            )

            runner.learn(num_learning_iterations=3)

            best = Path(runner.log_dir) / "model_best.pt"
            assert best.is_file()
            state = torch.load(best, weights_only=False)
            assert state["best_metric"] == runner.checkpoints.best_metric
            assert "best_iteration" in state

    def test_best_is_kept_even_between_scheduled_saves(self):
        """The best is checked every iteration, so it does not depend on save_interval."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = MockEnvironment(num_envs=4)
            env.max_episode_length = 2
            runner = OnPolicyRunner(
                env=env,
                # No periodic checkpoint falls inside three iterations except the final save
                runner_cfg=make_runner_cfg(log_dir=log_dir, save_interval=1000),
                device="cpu",
            )

            runner.learn(num_learning_iterations=3)

            assert (Path(runner.log_dir) / "model_best.pt").is_file()

    def test_resuming_from_best_loads_the_best_run(self):
        """resume_from="best" continues from the highest-scoring run of the experiment."""
        with tempfile.TemporaryDirectory() as log_dir:
            first = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir),
                device="cpu",
            )
            first.learn(num_learning_iterations=2)
            # Pin a score this run cannot be beaten on, and record it in its best checkpoint
            first.checkpoints.best_metric = None
            first.checkpoints.save_best(first._state_dict(), metric=1e6, iteration=99)

            resumed = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir, resume="best"),
                device="cpu",
            )

            assert resumed.resumed_from.name == "model_best.pt"
            assert resumed.current_learning_iteration == first.current_learning_iteration
            # The threshold comes along, so a weaker run does not overwrite the better checkpoint
            assert resumed.checkpoints.best_metric == 1e6

    def test_resuming_from_last_is_the_default(self):
        """Without asking, resuming continues from the most recent checkpoint."""
        with tempfile.TemporaryDirectory() as log_dir:
            first = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir),
                device="cpu",
            )
            first.learn(num_learning_iterations=2)

            resumed = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir, resume="last"),
                device="cpu",
            )

            assert resumed.resumed_from.name != "model_best.pt"
            assert resumed.resumed_from.name.startswith("model_")

    def test_a_worse_run_does_not_demote_the_best(self):
        """A resumed run that never beats the score it inherited leaves the best checkpoint alone.

        The threshold travels inside the checkpoint, so continuing from a strong policy does not
        reset the comparison and let a weaker run overwrite it.
        """
        with tempfile.TemporaryDirectory() as log_dir:
            first = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir),
                device="cpu",
            )
            first.learn(num_learning_iterations=2)
            # Pin a score the mock environment cannot reach, so any rewrite would be a regression
            first.checkpoints.best_metric = None
            best_path = first.checkpoints.save_best(
                first._state_dict(), metric=1e6, iteration=99
            )

            resumed = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir, resume="best"),
                device="cpu",
            )
            resumed.env.max_episode_length = 2
            resumed.learn(num_learning_iterations=3)

            assert resumed.checkpoints.best_metric == 1e6
            assert torch.load(best_path, weights_only=False)["best_metric"] == 1e6
            assert torch.load(best_path, weights_only=False)["best_iteration"] == 99


class _RenderableMockEnvironment(MockEnvironment):
    """A mock environment that renders, so the runner's video path can be exercised."""

    def __init__(self, render_mode: str | None = "rgb_array", **kwargs):
        super().__init__(**kwargs)
        self.render_mode = render_mode
        self.render_fps = 20.0
        self.renders = 0

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        self.renders += 1
        return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.mark.skipif(
    importlib.util.find_spec("imageio") is None, reason="imageio is not installed"
)
class TestVideoRecording:
    """Test the video the runner writes beside each checkpoint."""

    def test_a_clip_lands_beside_every_checkpoint(self):
        """Each checkpoint gets an MP4 of the rollout that produced it."""
        with tempfile.TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(
                env=_RenderableMockEnvironment(num_envs=2),
                runner_cfg=make_runner_cfg(
                    log_dir=log_dir, num_steps_per_env=4, save_interval=2, log_video=True
                ),
                device="cpu",
            )

            runner.learn(num_learning_iterations=3)

            run = Path(runner.log_dir)
            videos = sorted(path.name for path in run.glob("*.mp4"))
            assert videos, "no video was written"
            for video in videos:
                assert (run / video).with_suffix(".pt").is_file()

    def test_only_recorded_iterations_render(self):
        """Rendering happens on the iterations that checkpoint, not on every one."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = _RenderableMockEnvironment(num_envs=2)
            runner = OnPolicyRunner(
                env=env,
                runner_cfg=make_runner_cfg(
                    log_dir=log_dir, num_steps_per_env=4, save_interval=100, log_video=True
                ),
                device="cpu",
            )

            runner.learn(num_learning_iterations=3)

            # One probe render when the recorder is set up, to check the environment can render at
            # all, then four per recorded iteration: iteration 0 checkpoints, and so does the last
            assert env.renders == 1 + 4 + 4

    def test_nothing_is_recorded_without_the_flag(self):
        """A run that was not asked to record writes no video and never renders."""
        with tempfile.TemporaryDirectory() as log_dir:
            env = _RenderableMockEnvironment(num_envs=2)
            runner = OnPolicyRunner(
                env=env, runner_cfg=make_runner_cfg(log_dir=log_dir), device="cpu"
            )

            runner.learn(num_learning_iterations=2)

            assert runner.video is None
            assert env.renders == 0
            assert not list(Path(runner.log_dir).glob("*.mp4"))

    def test_an_environment_that_cannot_render_is_reported(self):
        """Asking for video from a headless environment warns instead of failing mid-run."""
        with tempfile.TemporaryDirectory() as log_dir:
            with patch("telekinesis.rlbotics.runner.logger") as mock_logger:
                runner = OnPolicyRunner(
                    env=_RenderableMockEnvironment(render_mode=None, num_envs=2),
                    runner_cfg=make_runner_cfg(log_dir=log_dir, log_video=True),
                    device="cpu",
                )

            assert runner.video is None
            assert mock_logger.warning.called
            assert "render_mode" in mock_logger.warning.call_args[0][0]

            # And training still runs
            runner.learn(num_learning_iterations=1)
            assert not list(Path(runner.log_dir).glob("*.mp4"))


class TestSymmetryWiring:
    """Test that the runner hands the environment to the symmetry extension."""

    @staticmethod
    def _mirror_recording(seen: dict):
        """Return a mirror function that records the environment it was called with."""

        def mirror(env=None, obs=None, actions=None):
            seen["env"] = env
            return (
                None if obs is None else torch.cat([obs, -obs]),
                None if actions is None else torch.cat([actions, -actions]),
            )

        return mirror

    def test_the_mirror_function_receives_the_training_environment(self):
        """The mirror function needs the environment to know the observation layout."""
        seen = {}
        env = MockEnvironment(num_envs=4)
        cfg = make_runner_cfg(num_steps_per_env=4)
        cfg.algorithm.symmetry_cfg = SymmetryConfig(
            data_augmentation_func=self._mirror_recording(seen), mirror_loss_coeff=0.5
        )

        runner = OnPolicyRunner(env=env, runner_cfg=cfg, device="cpu")
        runner.learn(num_learning_iterations=2)

        assert seen["env"] is env
        assert all(torch.isfinite(p).all() for p in runner.alg.actor.parameters())

    def test_symmetry_without_a_mirror_function_fails_at_config_time(self):
        """A symmetry config missing its function is rejected before a runner is ever built.

        The extension used to be a silent no-op, so this pins that a half-specified symmetry is an
        error rather than something that trains and quietly does nothing.
        """
        with pytest.raises(ValueError, match="data_augmentation_func"):
            SymmetryConfig(use_mirror_loss=True)

    def test_a_symmetry_dict_is_converted(self):
        """A plain dictionary is accepted, like every other nested config."""
        cfg = make_runner_cfg()
        cfg.algorithm = cfg.algorithm.replace(
            symmetry_cfg={"data_augmentation_func": "torch:tanh", "mirror_loss_coeff": 0.25}
        )

        assert isinstance(cfg.algorithm.symmetry_cfg, SymmetryConfig)
        assert cfg.algorithm.symmetry_cfg.mirror_loss_coeff == 0.25


@pytest.mark.skipif(not HAS_ONNX, reason="onnx is not installed")
class TestExportsTheBestPolicy:
    """Test that deployment gets the best policy, not whichever one training ended on."""

    @staticmethod
    def _pin_a_different_best(runner: OnPolicyRunner) -> torch.nn.Module:
        """Write a best checkpoint whose weights differ from the runner's current policy."""
        best_policy = runner.get_policy_snapshot(device="cpu")
        with torch.no_grad():
            for param in best_policy.parameters():
                param.mul_(0.0).add_(0.25)

        state = runner._state_dict()
        state["actor_state_dict"] = best_policy.state_dict()
        runner.checkpoints.best_metric = None
        runner.checkpoints.save_best(state, metric=1e6, iteration=99)
        return best_policy

    def _runner(self, log_dir: str) -> OnPolicyRunner:
        """Train briefly, with episodes short enough that a reward is scored."""
        env = MockEnvironment(num_envs=4)
        env.max_episode_length = 2
        runner = OnPolicyRunner(
            env=env,
            runner_cfg=make_runner_cfg(log_dir=log_dir, num_steps_per_env=4),
            device="cpu",
        )
        runner.learn(num_learning_iterations=3)
        return runner

    def test_the_export_carries_the_best_checkpoint(self):
        """The written graph computes what the best policy computes, not the final one."""
        pytest.importorskip("onnxruntime")
        from telekinesis.rlbotics.policy import Policy

        with tempfile.TemporaryDirectory() as log_dir:
            runner = self._runner(log_dir)
            best_policy = self._pin_a_different_best(runner)

            policy = Policy(runner.export())

            obs = torch.randn(3, 8)
            with torch.no_grad():
                expected = best_policy.as_onnx().eval()(obs)
                final = runner.get_policy_snapshot(device="cpu").as_onnx().eval()(obs)
            exported = torch.from_numpy(policy.get_action(obs.numpy()))

            assert torch.allclose(exported, expected, atol=1e-5)
            assert not torch.allclose(exported, final, atol=1e-5)

    def test_exporting_does_not_disturb_training(self):
        """Loading the best weights must not reach the actor being optimized."""
        with tempfile.TemporaryDirectory() as log_dir:
            runner = self._runner(log_dir)
            self._pin_a_different_best(runner)
            before = [p.clone() for p in runner.alg.get_policy().parameters()]

            runner.export()

            after = list(runner.alg.get_policy().parameters())
            assert all(torch.equal(a, b) for a, b in zip(before, after))
            assert runner.current_learning_iteration == 2

    def test_from_best_can_be_turned_off(self):
        """Exporting the final policy is still available, for comparing where training ended."""
        pytest.importorskip("onnxruntime")
        from telekinesis.rlbotics.policy import Policy

        with tempfile.TemporaryDirectory() as log_dir:
            runner = self._runner(log_dir)
            self._pin_a_different_best(runner)

            policy = Policy(runner.export(filename="final.onnx", from_best=False))

            obs = torch.randn(3, 8)
            with torch.no_grad():
                final = runner.get_policy_snapshot(device="cpu").as_onnx().eval()(obs)
            assert torch.allclose(torch.from_numpy(policy.get_action(obs.numpy())), final, atol=1e-5)

    def test_without_a_best_checkpoint_the_current_policy_is_exported(self):
        """A run that never scored anything still exports, rather than failing."""
        pytest.importorskip("onnxruntime")

        with tempfile.TemporaryDirectory() as log_dir:
            runner = OnPolicyRunner(
                env=MockEnvironment(num_envs=4),
                runner_cfg=make_runner_cfg(log_dir=log_dir, log_video=False),
                device="cpu",
            )

            assert runner.checkpoints.best() is None
            assert runner.export().is_file()
