"""Tests for the training logger."""

import importlib.util
import tempfile
from pathlib import Path

import pytest
import numpy as np
from tensordict import TensorDict
import torch

from telekinesis.rlbotics.config import LoggerConfig
from telekinesis.rlbotics.logger import Logger, VideoLogger


class TestLoggerInitialization:
    """Test Logger initialization."""

    def test_initialization_with_log_dir(self):
        """Test logger initialization with log directory."""
        with tempfile.TemporaryDirectory() as log_dir:
            logger = Logger(LoggerConfig(log_dir=log_dir), num_envs=4, device="cpu")

            # The logger puts the run in <log_dir>/<experiment>/<timestamp>
            assert logger.log_dir.startswith(log_dir)
            assert Path(logger.log_dir).parent.name == logger.cfg.experiment
            assert logger.num_envs == 4
            assert logger.device == "cpu"
            assert logger.writer is not None
            assert len(logger.reward_buffer) == 0
            assert len(logger.episode_length_buffer) == 0
            assert logger.total_timesteps == 0
            assert logger.total_time == 0.0

            logger.close()

    def test_initialization_without_log_dir(self):
        """Test logger initialization without log directory."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2, device="cpu")

        assert logger.log_dir is None
        assert logger.num_envs == 2
        assert logger.writer is None

    def test_context_manager(self):
        """Test logger as context manager."""
        with tempfile.TemporaryDirectory() as log_dir:
            with Logger(LoggerConfig(log_dir=log_dir), num_envs=2) as logger:
                assert logger.writer is not None
            # Logger should be closed after context exit


class TestProcessEnvStep:
    """Test environment step processing."""

    def test_process_single_step(self):
        """Test processing a single environment step."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=4, device="cpu")

        rewards = torch.randn(4)
        dones = torch.zeros(4)

        logger.process_env_step(rewards, dones)

        # No episodes should be completed
        assert len(logger.reward_buffer) == 0
        assert logger.cur_reward_sum[0] == rewards[0]
        assert logger.cur_episode_length[0] == 1

    def test_process_with_episode_done(self):
        """Test processing step with episode completion."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=4, device="cpu")

        # First step
        rewards = torch.ones(4)
        dones = torch.zeros(4)
        logger.process_env_step(rewards, dones)

        # Second step with done flags
        rewards = torch.ones(4)
        dones = torch.tensor([1.0, 0.0, 1.0, 0.0])
        logger.process_env_step(rewards, dones)

        # Two episodes should be completed
        assert len(logger.reward_buffer) == 2
        assert logger.reward_buffer[0] == 2.0  # 1 + 1
        assert logger.reward_buffer[1] == 2.0

    def test_process_with_episode_info(self):
        """Test processing with episode information."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2, device="cpu")

        rewards = torch.randn(2)
        dones = torch.zeros(2)
        episode_info = {"goal_reached": True, "steps": 10}

        logger.process_env_step(rewards, dones, episode_info)

        assert len(logger.episode_info_buffer) == 1
        assert logger.episode_info_buffer[0]["goal_reached"] is True

    def test_reward_buffer_maxlen(self):
        """Test that reward buffer respects maxlen."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=1, device="cpu")

        # Add more than maxlen episodes
        for i in range(150):
            rewards = torch.tensor([float(i)])
            dones = torch.tensor([1.0])
            logger.process_env_step(rewards, dones)

        # Buffer should only contain last 100
        assert len(logger.reward_buffer) == 100

    def test_episode_length_tracking(self):
        """Test episode length tracking."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=1, device="cpu")

        # Simulate 5-step episode
        for step in range(5):
            rewards = torch.tensor([1.0])
            dones = torch.tensor([1.0 if step == 4 else 0.0])
            logger.process_env_step(rewards, dones)

        assert len(logger.episode_length_buffer) == 1
        assert logger.episode_length_buffer[0] == 5


class TestLogging:
    """Test logging functionality."""

    def test_log_with_metrics(self):
        """Test logging with various metrics."""
        with tempfile.TemporaryDirectory() as log_dir:
            logger = Logger(LoggerConfig(log_dir=log_dir), num_envs=4)

            # Add some episode data
            for _ in range(5):
                rewards = torch.ones(4)
                dones = torch.ones(4)
                logger.process_env_step(rewards, dones)

            # Log metrics
            losses = {"policy": 0.5, "value": 0.3}
            logger.log(
                iteration=0,
                total_iterations=10,
                losses=losses,
                learning_rate=0.001,
                action_std=torch.tensor([1.0]),
                custom_metrics={"ratio": 0.9},
                print_interval=1,
            )

            assert logger.total_timesteps == 20  # 4 envs * 5 steps
            assert logger.writer is not None

            logger.close()

    def test_log_without_writer(self):
        """Test logging without TensorBoard writer."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        # Should not raise error even without writer
        logger.log(
            iteration=0,
            total_iterations=1,
            losses={"loss": 0.5},
        )

    def test_log_with_tensor_metrics(self):
        """Test logging with tensor metrics."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        losses = {
            "policy": torch.tensor(0.5),
            "value": torch.tensor(0.3),
        }

        action_std = torch.tensor([1.0, 0.9])

        # Should handle tensors correctly
        logger.log(
            iteration=0,
            total_iterations=1,
            losses=losses,
            action_std=action_std,
        )

    def test_log_fps_calculation(self):
        """Test FPS calculation during logging."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=10)

        # Simulate some steps
        for _ in range(5):
            rewards = torch.randn(10)
            dones = torch.zeros(10)
            logger.process_env_step(rewards, dones)

        # Log with timing
        logger.log(
            iteration=0,
            total_iterations=1,
            collect_time=0.1,
            learn_time=0.05,
        )

        assert logger.total_timesteps == 50


class TestMetricsBuffers:
    """Test metrics buffer management."""

    def test_reward_buffer(self):
        """Test reward buffer functionality."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        # Add several episodes
        episodes = [10.5, 15.2, 12.8, 11.0]
        for reward_sum in episodes:
            logger.reward_buffer.append(reward_sum)

        assert len(logger.reward_buffer) == 4
        assert logger.reward_buffer[0] == 10.5

    def test_episode_info_buffer(self):
        """Test episode info buffer."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=1)

        info1 = {"success": True, "steps": 10}
        info2 = {"success": False, "steps": 20}

        logger.episode_info_buffer.append(info1)
        logger.episode_info_buffer.append(info2)

        assert len(logger.episode_info_buffer) == 2
        assert logger.episode_info_buffer[0]["success"] is True

    def test_running_statistics(self):
        """Test running episode statistics."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=3)

        # Simulate multi-step episodes
        for step in range(10):
            rewards = torch.tensor([1.0, 2.0, 1.5])
            dones = torch.zeros(3)
            logger.process_env_step(rewards, dones)

        # Check running statistics
        expected_sums = torch.tensor([10.0, 20.0, 15.0])
        assert torch.allclose(logger.cur_reward_sum, expected_sums)
        assert torch.allclose(
            logger.cur_episode_length, torch.tensor([10.0, 10.0, 10.0])
        )


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_squeeze_rewards_dim(self):
        """Test squeezing reward dimensions."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        # 2D rewards
        rewards = torch.randn(2, 1)
        dones = torch.zeros(2)

        logger.process_env_step(rewards, dones)

        # Should handle shape correctly
        assert logger.cur_reward_sum.shape == (2,)

    def test_empty_reward_buffer_stats(self):
        """Test statistics with empty buffer."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        # Log without any episodes completed
        logger.log(
            iteration=0,
            total_iterations=1,
        )

        # Should not crash with empty buffer

    def test_custom_metrics_types(self):
        """Test custom metrics with different types."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        custom_metrics = {
            "float_metric": 0.5,
            "tensor_metric": torch.tensor(0.3),
            "int_metric": 10,
        }

        logger.log(
            iteration=0,
            total_iterations=1,
            custom_metrics=custom_metrics,
        )

    def test_zero_iteration_time(self):
        """Test logging with zero iteration time."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2)

        logger.log(
            iteration=0,
            total_iterations=1,
            collect_time=0.0,
            learn_time=0.0,
        )

        # Should handle zero time gracefully

    def test_no_episodes_completed(self):
        """Test logging when no episodes are completed."""
        logger = Logger(LoggerConfig(log_dir=None), num_envs=4)

        # Single step with no dones
        rewards = torch.ones(4)
        dones = torch.zeros(4)
        logger.process_env_step(rewards, dones)

        # Log should work with empty buffers
        logger.log(
            iteration=0,
            total_iterations=1,
            losses={"loss": 0.5},
        )

        assert len(logger.reward_buffer) == 0


class _RenderableEnv:
    """Minimal stand-in for a renderable environment, enough for VideoLogger to drive."""

    def __init__(self, render_mode: str | None = "rgb_array", num_envs: int = 1):
        self.num_envs = num_envs
        self.num_actions = 2
        self.render_mode = render_mode
        self.render_fps = 25.0
        self.max_episode_length = 4
        self.steps = 0
        self.closed = False

    def _obs(self) -> TensorDict:
        return TensorDict(
            {"observation": torch.zeros(self.num_envs, 3)}, batch_size=(self.num_envs,)
        )

    def get_observations(self) -> TensorDict:
        return self._obs()

    def reset(self) -> TensorDict:
        return self._obs()

    def step(self, actions):
        self.steps += 1
        return self._obs(), torch.zeros(self.num_envs), torch.zeros(self.num_envs), {}

    def render(self):
        if self.render_mode != "rgb_array":
            return (None,) * self.num_envs
        # A vectorized environment renders one frame per sub-environment
        return tuple(np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(self.num_envs))

    def close(self):
        self.closed = True


needs_imageio = pytest.mark.skipif(
    importlib.util.find_spec("imageio") is None, reason="imageio is not installed"
)


class TestVideoLogger:
    """Test recording a policy as video."""

    def test_fps_comes_from_the_environment(self):
        """The playback rate matches what the simulation renders at, unless told otherwise."""
        assert VideoLogger(_RenderableEnv()).fps == 25.0
        assert VideoLogger(_RenderableEnv(), fps=10.0).fps == 10.0

    def test_frame_unwraps_the_vector_tuple(self):
        """A vectorized environment renders a frame per sub-environment; the first one is taken."""
        frame = VideoLogger(_RenderableEnv(num_envs=4)).frame()

        assert frame.shape == (8, 8, 3)

    def test_can_render_is_false_without_a_render_mode(self):
        """An environment built headless reports that it cannot be recorded, rather than failing."""
        assert not VideoLogger(_RenderableEnv(render_mode=None)).can_render
        assert VideoLogger(_RenderableEnv()).can_render

    @needs_imageio
    def test_write_produces_a_file(self, tmp_path):
        """Frames handed over directly are written, which is the path the runner uses."""
        recorder = VideoLogger(_RenderableEnv())
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

        path = recorder.write(frames, tmp_path / "clip.mp4")

        assert path.is_file()
        assert path.stat().st_size > 0

    @needs_imageio
    def test_record_rolls_the_policy_out(self, tmp_path):
        """Recording steps the environment itself, deterministically, for one episode."""
        env = _RenderableEnv()
        recorder = VideoLogger(env, obs_groups="observation")
        calls = []

        def policy(obs, stochastic=True):
            calls.append(stochastic)
            return torch.zeros(env.num_envs, env.num_actions)

        path = recorder.record(policy, tmp_path / "rollout.mp4")

        assert path.is_file()
        # One step per frame after the initial reset frame, and never sampled
        assert env.steps == env.max_episode_length
        assert calls == [False] * env.max_episode_length

    @needs_imageio
    def test_record_honours_num_steps(self, tmp_path):
        """A shorter clip can be asked for at the call site."""
        env = _RenderableEnv()

        VideoLogger(env).record(lambda obs, stochastic=True: torch.zeros(1, 2), tmp_path / "s.mp4", num_steps=2)

        assert env.steps == 2

    def test_several_observation_groups_are_concatenated(self):
        """An actor reading more than one group gets them joined, as the runner does."""
        env = _RenderableEnv()
        env._obs = lambda: TensorDict(
            {"a": torch.zeros(1, 3), "b": torch.ones(1, 2)}, batch_size=(1,)
        )
        recorder = VideoLogger(env, obs_groups=["a", "b"])

        assert recorder._actor_obs(env.get_observations()).shape == (1, 5)

    def test_close_closes_the_environment(self):
        """The recorder owns its environment, so closing it reaches through."""
        env = _RenderableEnv()

        VideoLogger(env).close()

        assert env.closed


class TestRunDirectory:
    """Test where a run writes, which has to be its own directory."""

    def test_two_runs_started_together_do_not_share_a_directory(self, tmp_path):
        """The timestamp resolves to the second, so back-to-back runs need a suffix to stay apart.

        Sharing one would interleave two runs' checkpoints and let one's rotation delete the other's.
        """
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp")

        first = Logger(cfg, num_envs=1)
        second = Logger(cfg, num_envs=1)
        third = Logger(cfg, num_envs=1)

        assert len({first.log_dir, second.log_dir, third.log_dir}) == 3
        assert Path(second.log_dir).name.startswith(Path(first.log_dir).name)
        for logger in (first, second, third):
            logger.close()
