"""Example: record a video of a checkpoint with VideoLogger.

Training writes an MP4 beside every checkpoint on its own, given ``LoggerConfig(log_video=True)`` and
an environment built to render. This is the other half: taking a checkpoint that already exists and
looking at what it does, which is what you want after a run has finished and one checkpoint looks
more interesting than the rest.

Four steps: build a renderable environment, train briefly to get a checkpoint, load that checkpoint
into a fresh runner, record it. The environment here is a toy that draws its own frames with numpy,
so the example needs no simulator::

    python examples/module_examples/video_example.py

Needs imageio to write the file: pip install "telekinesis-rlbotics[examples]"
"""

import importlib.util
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
)
from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.logger import VideoLogger
from telekinesis.rlbotics.runner import OnPolicyRunner

# Exit code that tells run_all_examples.py this example was skipped, not that it failed
SKIPPED = 2

FRAME_SIZE = 96


class ReachTheCentreEnv(VecEnv):
    """A toy environment that renders itself: push a dot towards the middle of the frame.

    Small enough to train in seconds and drawn with numpy rather than a physics engine, so what this
    example demonstrates is the recording, not the simulator. The contract is the same one every real
    adapter implements: observations as a TensorDict, rewards and dones shaped ``(num_envs,)``, and
    ``render()`` returning frames when the environment was built to render.
    """

    def __init__(self, num_envs: int = 8, render_mode: str | None = None) -> None:
        """Create the environment.

        Args:
            num_envs: Number of environments stepped in parallel.
            render_mode: "rgb_array" to draw frames, or None to render nothing, which is what a
                training environment normally does.
        """
        self.num_envs = num_envs
        self.num_actions = 2
        self.device = torch.device("cpu")
        self.max_episode_length = 60
        self.render_mode = render_mode
        self.render_fps = 20.0
        self.position = torch.zeros(num_envs, 2)
        self.step_count = 0
        self.reset()

    def _observations(self) -> TensorDict:
        """Wrap the positions as observations."""
        return TensorDict({"observation": self.position.clone()}, batch_size=(self.num_envs,))

    def get_observations(self) -> TensorDict:
        """Return the current observations without stepping."""
        return self._observations()

    def reset(self) -> TensorDict:
        """Scatter the dots and start a new episode."""
        self.position = torch.rand(self.num_envs, 2) * 2.0 - 1.0
        self.step_count = 0
        return self._observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Move the dots, reward being close to the centre, and end the episode on the time limit.

        Args:
            actions: Nudges to apply, one row per environment.

        Returns:
            Observations, rewards, dones and extras, per the VecEnv contract.
        """
        self.position = (self.position + 0.1 * actions.clamp(-1.0, 1.0)).clamp(-1.0, 1.0)
        self.step_count += 1

        rewards = -self.position.norm(dim=-1)
        timed_out = self.step_count >= self.max_episode_length
        dones = torch.full((self.num_envs,), float(timed_out))
        extras = {"time_outs": torch.full((self.num_envs,), timed_out, dtype=torch.bool)}
        if timed_out:
            self.reset()
        return self._observations(), rewards, dones, extras

    def render(self) -> np.ndarray | None:
        """Draw the first environment's dot, with the target it is aiming for.

        Returns:
            An RGB frame, or None when the environment was not built to render, which is what the
            runner checks before it starts recording.
        """
        if self.render_mode != "rgb_array":
            return None

        frame = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        centre = FRAME_SIZE // 2
        # The target: a dim cross in the middle
        frame[centre - 1 : centre + 1, :, :] = 40
        frame[:, centre - 1 : centre + 1, :] = 40

        # The dot, mapped from [-1, 1] onto the frame
        row, column = ((self.position[0] * 0.5 + 0.5) * (FRAME_SIZE - 8) + 4).round().int().tolist()
        frame[row - 3 : row + 3, column - 3 : column + 3] = (80, 220, 120)
        return frame

    def close(self) -> None:
        """Nothing to release."""


def make_runner_cfg(log_dir: str, resume: str | None = None) -> OnPolicyRunnerConfig:
    """Build a small training configuration.

    Args:
        log_dir: Where the run writes its checkpoints.
        resume: Checkpoint to continue from, or None to train from scratch.

    Returns:
        The runner configuration.
    """
    return OnPolicyRunnerConfig(
        obs_groups={"actor": ["observation"], "critic": ["observation"]},
        num_steps_per_env=60,
        verbose=False,
        logger=LoggerConfig(
            log_dir=log_dir, experiment="video_example", save_interval=10, resume=resume
        ),
        algorithm=PPOConfig(learning_rate=3e-3, num_learning_epochs=5, num_mini_batches=1),
        actor=MLPConfig(
            hidden_dims=(32, 32), distribution_cfg=GaussianDistributionConfig(init_std=0.5)
        ),
        critic=MLPConfig(hidden_dims=(32, 32)),
    )


def main() -> int:
    """Train, then load a checkpoint and record it.

    Returns:
        Process exit code: 0 on success, or SKIPPED when imageio is not installed.
    """
    if importlib.util.find_spec("imageio") is None:
        logger.warning(
            'skipping: writing an MP4 needs imageio. pip install "telekinesis-rlbotics[examples]"'
        )
        return SKIPPED

    with tempfile.TemporaryDirectory() as log_dir:
        # 1. Train briefly, on an environment that renders nothing. This is the normal case: a
        #    training run is headless, and the checkpoints are what it leaves behind
        runner = OnPolicyRunner(
            env=ReachTheCentreEnv(num_envs=8), runner_cfg=make_runner_cfg(log_dir), device="cpu"
        )
        runner.learn(num_learning_iterations=20)
        checkpoint = runner.checkpoints.best() or runner.checkpoints.latest()
        logger.info(f"1. trained, best checkpoint at {checkpoint.name}")

        # 2. Load that checkpoint into a fresh runner. Resuming takes "best", "last", a file name or
        #    a path, so this is the same code path training uses to continue a run
        replay = OnPolicyRunner(
            env=ReachTheCentreEnv(num_envs=1),
            runner_cfg=make_runner_cfg(log_dir, resume=str(checkpoint)),
            device="cpu",
        )
        logger.info(f"2. loaded {replay.resumed_from.name} at iteration {replay.current_learning_iteration}")

        # 3. Record it. The recording environment is its own, built to render and with a single
        #    sub-environment, and get_policy_snapshot() hands over a copy so nothing being trained
        #    is disturbed
        video = VideoLogger(
            ReachTheCentreEnv(num_envs=1, render_mode="rgb_array"), obs_groups="observation"
        )
        logger.info(f"3. recording at {video.fps} fps, can_render={video.can_render}")

        path = video.record(replay.get_policy_snapshot(), Path(log_dir) / "policy.mp4")
        video.close()

        size = path.stat().st_size
        logger.info(f"4. wrote {path.name}, {size / 1024:.1f} KiB, {video.num_steps or 60} frames")
        logger.info("   During training the runner does this itself: LoggerConfig(log_video=True)")
        logger.info("   plus an environment built with render_mode='rgb_array'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
