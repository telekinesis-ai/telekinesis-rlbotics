"""Training metrics logging to TensorBoard, and video recording."""

from __future__ import annotations

import datetime
import json
import os
import statistics
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import torch
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from telekinesis.rlbotics.config import LoggerConfig


class Logger:
    """Logger for training metrics with TensorBoard support."""

    def __init__(
        self,
        cfg: LoggerConfig | None = None,
        num_envs: int = 1,
        num_steps_per_env: int = 1,
        device: str = "cpu",
    ) -> None:
        """Initialize the logger from its config.

        Args:
            cfg: Logging configuration, which decides where the run is written. Defaults to None,
                which uses the defaults.
            num_envs: Number of parallel environments.
            num_steps_per_env: Environment steps collected per environment per iteration, used to
                turn an iteration's wall-clock time into a steps/second FPS figure.
            device: Device for tensors ("cpu" or "cuda").
        """
        self.cfg = cfg if cfg is not None else LoggerConfig()
        self.num_envs = num_envs
        self.num_steps_per_env = num_steps_per_env
        self.device = device

        # One directory per run: <log_dir>/<experiment>/<timestamp>, with everything flat inside
        if self.cfg.enabled and self.cfg.experiment_dir is not None:
            self.log_dir = self._unique_run_dir(self.cfg.experiment_dir)
        else:
            self.log_dir = None

        # Initialize TensorBoard writer. It is only created when logging is on, a directory was
        # given, and metrics are wanted, because the writer is what creates the event file.
        if self.log_dir is not None and self.cfg.log_metrics:
            os.makedirs(self.log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        else:
            self.writer = None

        # Metrics buffers
        self._reward_buffer = deque(maxlen=100)
        self._episode_length_buffer = deque(maxlen=100)
        self.episode_info_buffer = []

        # Completed-episode rewards/lengths, kept on-device until _flush_pending_episodes()
        # pulls them to the CPU in one batched sync instead of per-step.
        self._pending_rewards: list[torch.Tensor] = []
        self._pending_lengths: list[torch.Tensor] = []

        # Running episode statistics
        self.cur_reward_sum = torch.zeros(num_envs, dtype=torch.float32, device=device)
        self.cur_episode_length = torch.zeros(num_envs, dtype=torch.float32, device=device)

        # Timing statistics
        self.total_timesteps = 0
        self.total_time = 0.0

    @staticmethod
    def _unique_run_dir(experiment_dir: str) -> str:
        """Return a fresh run directory, timestamped and never an existing one.

        The timestamp resolves to the second, so two runs launched back to back — a sweep, or a
        resume script — would otherwise share a directory and interleave their checkpoints. A
        suffix keeps them apart while leaving the common case readable.

        Args:
            experiment_dir: Directory holding the experiment's runs.

        Returns:
            The path of the run directory to use.
        """
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        candidate = os.path.join(experiment_dir, stamp)
        suffix = 1
        while os.path.exists(candidate):
            candidate = os.path.join(experiment_dir, f"{stamp}_{suffix}")
            suffix += 1
        return candidate

    @property
    def reward_buffer(self) -> deque:
        """Rolling buffer of completed-episode rewards, synced from the GPU on access."""
        self._flush_pending_episodes()
        return self._reward_buffer

    @property
    def episode_length_buffer(self) -> deque:
        """Rolling buffer of completed-episode lengths, synced from the GPU on access."""
        self._flush_pending_episodes()
        return self._episode_length_buffer

    @property
    def mean_reward(self) -> float | None:
        """Mean return of the recently completed episodes, or None if none have finished yet.

        This is the number the console and the event file report as the reward, and what the
        checkpoint manager scores a policy on.
        """
        rewards = self.reward_buffer
        return statistics.mean(rewards) if rewards else None

    def save_config(self, config: dict) -> str | None:
        """Write the resolved configuration next to the logs, for reproducing the run.

        Args:
            config: The configuration as a plain dictionary.

        Returns:
            The path written, or None if there is nothing to write to or ``log_config`` is off.
        """
        if not (self.cfg.enabled and self.cfg.log_config) or self.log_dir is None:
            return None

        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "config.json")
        with open(path, "w") as handle:
            json.dump(config, handle, indent=2, default=str)
        return path

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        episode_infos: dict | None = None,
    ) -> None:
        """Process environment step and update metrics.

        Args:
            rewards: Reward tensor of shape (num_envs,) or (num_envs, 1).
            dones: Done flags of shape (num_envs,) or (num_envs, 1).
            episode_infos: Optional dict with episode information to log.
        """
        # Ensure correct shapes
        if rewards.dim() > 1:
            rewards = rewards.squeeze(-1)
        if dones.dim() > 1:
            dones = dones.squeeze(-1)

        # Update cumulative rewards and episode lengths
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1

        # Update total timesteps (count every step across all environments)
        self.total_timesteps += len(rewards)

        # Store episode info if provided (always track, regardless of writer)
        if episode_infos is not None:
            self.episode_info_buffer.append(episode_infos)

        # Check for episode completions. Kept as a GPU tensor op (no .item()/.cpu() here) so this
        # doesn't force a host sync on every rollout step.
        done_indices = (dones > 0).nonzero(as_tuple=False).squeeze(-1)

        if done_indices.numel() > 0:
            # Stash completed episode metrics on-device; _flush_pending_episodes() batches the
            # actual CPU sync so it only happens when the buffers are read (e.g. at log time).
            self._pending_rewards.append(self.cur_reward_sum[done_indices].clone())
            self._pending_lengths.append(self.cur_episode_length[done_indices].clone())

            # Reset completed episodes
            self.cur_reward_sum[done_indices] = 0
            self.cur_episode_length[done_indices] = 0

    def _flush_pending_episodes(self) -> None:
        """Move any pending completed-episode stats from GPU to the CPU-side buffers."""
        if not self._pending_rewards:
            return

        rewards = torch.cat(self._pending_rewards).cpu().numpy().tolist()
        lengths = torch.cat(self._pending_lengths).cpu().numpy().tolist()
        self._reward_buffer.extend(rewards)
        self._episode_length_buffer.extend(lengths)
        self._pending_rewards.clear()
        self._pending_lengths.clear()

    def log(
        self,
        iteration: int,
        total_iterations: int,
        start_iteration: int = 0,
        collect_time: float = 0.0,
        learn_time: float = 0.0,
        losses: dict | None = None,
        learning_rate: float = 0.0,
        action_std: torch.Tensor | None = None,
        custom_metrics: dict | None = None,
        diagnostics: dict | None = None,
        print_interval: int | None = None,
    ) -> None:
        """Log training metrics to TensorBoard and console.

        Args:
            iteration: Current iteration number.
            total_iterations: Total iterations for training.
            start_iteration: Starting iteration for ETA calculation.
            collect_time: Time spent collecting environment transitions.
            learn_time: Time spent on learning updates.
            losses: Dictionary of loss values.
            learning_rate: Current learning rate.
            action_std: Action standard deviation tensor.
            custom_metrics: Dictionary of custom metrics to log.
            diagnostics: Dictionary of training diagnostics, such as the policy KL divergence,
                the PPO clip fraction and the value function's explained variance.
            print_interval: Iterations between console reports. Defaults to None, which uses
                the config's ``log_interval``.
            print_interval: Interval for printing to console.
        """
        # Update timing statistics (always track)
        iteration_time = collect_time + learn_time
        self.total_time += iteration_time

        if not self.cfg.enabled:
            return
        if print_interval is None:
            print_interval = self.cfg.log_interval

        if self.writer is None:
            # Only print, don't log to TensorBoard
            if print_interval and iteration % print_interval == 0:
                self._print_log(
                    iteration,
                    total_iterations,
                    start_iteration,
                    collect_time,
                    learn_time,
                    losses,
                    learning_rate,
                    action_std,
                    custom_metrics,
                    diagnostics,
                )
            return

        # Log to TensorBoard
        # Performance metrics
        if iteration_time > 0:
            fps = int(self.num_envs * self.num_steps_per_env / iteration_time)
            self.writer.add_scalar("Performance/FPS", fps, iteration)
        self.writer.add_scalar("Performance/collect_time", collect_time, iteration)
        self.writer.add_scalar("Performance/learn_time", learn_time, iteration)

        # Episode metrics
        if len(self.reward_buffer) > 0:
            mean_reward = statistics.mean(self.reward_buffer)
            mean_length = statistics.mean(self.episode_length_buffer)
            self.writer.add_scalar("Episodes/mean_reward", mean_reward, iteration)
            self.writer.add_scalar("Episodes/mean_length", mean_length, iteration)

        # Loss metrics
        if losses is not None:
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"Loss/{key}", value, iteration)

        # Learning rate
        if learning_rate > 0:
            self.writer.add_scalar("Learning/learning_rate", learning_rate, iteration)

        # Action standard deviation
        if action_std is not None:
            if isinstance(action_std, torch.Tensor):
                std_val = action_std.mean().item()
            else:
                std_val = action_std
            self.writer.add_scalar("Policy/action_std", std_val, iteration)

        # Custom metrics
        if custom_metrics is not None:
            for key, value in custom_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"Custom/{key}", value, iteration)

        # Training diagnostics
        if diagnostics is not None:
            for key, value in diagnostics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                self.writer.add_scalar(f"Diagnostics/{key}", value, iteration)

        # Print to console
        if print_interval and iteration % print_interval == 0:
            self._print_log(
                iteration,
                total_iterations,
                start_iteration,
                collect_time,
                learn_time,
                losses,
                learning_rate,
                action_std,
                custom_metrics,
                diagnostics,
            )

    def _print_log(
        self,
        iteration: int,
        total_iterations: int,
        start_iteration: int,
        collect_time: float,
        learn_time: float,
        losses: dict | None,
        learning_rate: float,
        action_std: torch.Tensor | None,
        custom_metrics: dict | None,
        diagnostics: dict | None = None,
    ) -> None:
        """Print log to console."""
        width = 80
        pad = 35

        log_string = f"{'#' * width}\n"
        log_string += (
            f"\033[1m{f' Iteration {iteration}/{total_iterations} '.center(width)}\033[0m\n\n"
        )

        # Performance metrics
        iteration_time = collect_time + learn_time
        if iteration_time > 0:
            fps = int(self.num_envs * self.num_steps_per_env / iteration_time)
            log_string += f"{'FPS:':>{pad}} {fps}\n"
        log_string += f"{'Total timesteps:':>{pad}} {self.total_timesteps}\n"
        log_string += f"{'Collection time:':>{pad}} {collect_time:.3f}s\n"
        log_string += f"{'Learning time:':>{pad}} {learn_time:.3f}s\n"

        # Episode metrics
        if len(self.reward_buffer) > 0:
            mean_reward = statistics.mean(self.reward_buffer)
            mean_length = statistics.mean(self.episode_length_buffer)
            log_string += f"{'Mean reward:':>{pad}} {mean_reward:.4f}\n"
            log_string += f"{'Mean episode length:':>{pad}} {mean_length:.1f}\n"

        # Loss metrics
        if losses is not None:
            for key, value in losses.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                log_string += f"{f'Mean {key} loss:':>{pad}} {value:.6f}\n"

        # Learning rate
        if learning_rate > 0:
            log_string += f"{'Learning rate:':>{pad}} {learning_rate:.6f}\n"

        # Action standard deviation
        if action_std is not None:
            if isinstance(action_std, torch.Tensor):
                std_val = action_std.mean().item()
            else:
                std_val = action_std
            log_string += f"{'Action std:':>{pad}} {std_val:.4f}\n"

        # Custom metrics
        if custom_metrics is not None:
            for key, value in custom_metrics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                log_string += f"{f'{key}:':>{pad}} {value:.6f}\n"

        # Training diagnostics
        if diagnostics is not None:
            labels = {
                "kl": "KL divergence",
                "clip_fraction": "Clip fraction",
                "explained_variance": "Explained variance",
            }
            for key, value in diagnostics.items():
                if isinstance(value, torch.Tensor):
                    value = value.item()
                label = labels.get(key, key)
                log_string += f"{f'{label}:':>{pad}} {value:.4f}\n"

        # Time estimates
        log_string += f"{'-' * width}\n"
        elapsed_time = datetime.timedelta(seconds=int(self.total_time))
        log_string += f"{'Time elapsed:':>{pad}} {elapsed_time}\n"

        # ETA calculation
        done_iterations = iteration + 1 - start_iteration
        remaining_iterations = total_iterations - start_iteration - done_iterations
        if done_iterations > 0:
            eta_seconds = (self.total_time / done_iterations) * remaining_iterations
            eta_time = datetime.timedelta(seconds=int(eta_seconds))
            log_string += f"{'ETA:':>{pad}} {eta_time}\n"

        log_string += f"{'#' * width}\n"
        print(log_string)

    def close(self) -> None:
        """Close the logger and flush TensorBoard writer."""
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

class VideoLogger:
    """Records what a policy looks like, as an MP4 beside the checkpoint it belongs to.

    Reward curves say a policy improved; a video says whether it is walking or shuffling on one
    knee, which is why the runner writes one with every checkpoint when ``LoggerConfig.log_video``
    is on. Set up by :class:`~telekinesis.rlbotics.runner.OnPolicyRunner`, so a training script only
    has to build its environment with ``render_mode="rgb_array"`` and switch the flag on.

    Two ways in. The runner uses :meth:`frame` and :meth:`write`, capturing the rollout it is already
    performing, so recording costs one render per step and no extra simulation. A script with a
    checkpoint in hand uses :meth:`record`, which drives its own rollout of a policy.

    Example:
        Record a checkpoint's policy after training::

            env = GymnasiumVecEnv("Hopper-v5", 1, "cpu", render_mode="rgb_array")
            runner.load("logs/exp/run/model_best.pt")
            VideoLogger(env).record(runner.get_policy_snapshot(), "best.mp4")
    """

    def __init__(
        self,
        env,
        obs_groups: str | Iterable[str] = "observation",
        fps: float | None = None,
        num_steps: int | None = None,
    ) -> None:
        """Initialize the recorder.

        Args:
            env: The environment to render. It has to have been built to render, which for every
                adapter in :mod:`telekinesis.rlbotics.envs` means ``render_mode="rgb_array"``.
            obs_groups: Observation group feeding the policy, or several to concatenate, matching the
                runner's ``obs_groups["actor"]``. Only :meth:`record` uses it. Defaults to
                "observation".
            fps: Playback rate. Defaults to None, which asks the environment for ``render_fps``.
            num_steps: Steps :meth:`record` rolls out. Defaults to None, which uses the environment's
                ``max_episode_length``, so one episode.
        """
        self.env = env
        self.obs_groups = [obs_groups] if isinstance(obs_groups, str) else list(obs_groups)
        self.fps = fps if fps is not None else float(getattr(env, "render_fps", 30.0))
        self.num_steps = num_steps

    @property
    def can_render(self) -> bool:
        """Whether the environment actually produces frames, so a caller can say so up front."""
        return self.frame() is not None

    def frame(self):
        """Return one rendered frame, or None if the environment was not built to render.

        Returns:
            An RGB frame shaped (H, W, 3), taking the first sub-environment of a vectorized one, or
            None.
        """
        frame = self.env.render()
        if isinstance(frame, (tuple, list)):
            frame = frame[0] if frame else None
        return frame

    def write(self, frames: Iterable, path: str | Path) -> Path:
        """Write already-captured frames to an MP4.

        Args:
            frames: RGB frames, each shaped (H, W, 3). Consumed lazily, so a generator that also
                drives a rollout works and is never left half-drained.
            path: File to write to. Parent directories are created if missing.

        Returns:
            The path written.

        Raises:
            ImportError: If imageio is not installed.
        """
        import imageio

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(path), fps=self.fps)
        try:
            for frame in frames:
                writer.append_data(frame)
        finally:
            writer.close()
        return path

    def record(self, policy, path: str | Path, num_steps: int | None = None) -> Path:
        """Roll a policy out in this environment and write the frames.

        Runs deterministically and under ``torch.inference_mode()``: a video should show the policy's
        mean action, and a simulator stepped during training holds buffers that cannot be updated in
        place outside that mode.

        Args:
            policy: The policy to record, called as ``policy(obs, stochastic=False)``.
            path: File to write to.
            num_steps: Steps to record. Defaults to None, which uses the value given at construction.

        Returns:
            The path written.
        """
        steps = num_steps or self.num_steps or int(getattr(self.env, "max_episode_length", 200))

        def frames():
            with torch.inference_mode():
                obs = self.env.reset()
                yield self.frame()
                for _ in range(steps):
                    action = policy(self._actor_obs(obs), stochastic=False)
                    obs = self.env.step(action)[0]
                    yield self.frame()

        return self.write(frames(), path)

    def _actor_obs(self, obs: TensorDict) -> torch.Tensor:
        """Assemble the policy's input from the observation groups it reads.

        Args:
            obs: Observations from the environment.

        Returns:
            The actor's observation tensor.
        """
        if len(self.obs_groups) == 1:
            return obs[self.obs_groups[0]]
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def close(self) -> None:
        """Close the environment being recorded in."""
        self.env.close()
