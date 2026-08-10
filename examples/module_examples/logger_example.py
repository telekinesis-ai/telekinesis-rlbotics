"""Example: TensorBoard logger for training.

This example demonstrates:
- Creating a logger with TensorBoard support
- Processing environment steps
- Logging training metrics
- Tracking episode rewards and lengths
- Custom metrics logging
- Checkpoint saving
"""

import tempfile

import torch
from loguru import logger as loguru_logger

from telekinesis.rlbotics.checkpoint import CheckpointManager
from telekinesis.rlbotics.config import LoggerConfig
from telekinesis.rlbotics.logger import Logger


def main():
    """Demonstrate logger usage."""
    loguru_logger.info("TensorBoard Logger Example")
    loguru_logger.info("=" * 60)

    # Create temporary directory for logs
    with tempfile.TemporaryDirectory() as log_dir:
        # 1. Initialize Logger
        loguru_logger.info("1. Initializing Logger")
        loguru_logger.info("-" * 60)

        num_envs = 4
        device = "cpu"

        logger = Logger(
            LoggerConfig(log_dir=log_dir, log_interval=1),
            num_envs=num_envs,
            device=device,
        )

        loguru_logger.info(f"   Log directory: {log_dir}")
        loguru_logger.info(f"   Number of environments: {num_envs}")
        loguru_logger.info(f"   Device: {device}")

        # 2. Simulate Training Loop
        loguru_logger.info("2. Simulating Training Loop")
        loguru_logger.info("-" * 60)

        torch.manual_seed(42)
        total_iterations = 10
        steps_per_iteration = 5

        for iteration in range(total_iterations):
            loguru_logger.info(f"   Iteration {iteration + 1}/{total_iterations}")

            # Simulate environment steps
            for step in range(steps_per_iteration):
                # Random rewards and dones
                rewards = torch.randn(num_envs, device=device) * 10
                dones = (torch.rand(num_envs, device=device) < 0.1).float()

                # Optional episode info
                episode_info = {
                    "step": step,
                    "iteration": iteration,
                } if step == 0 else None

                logger.process_env_step(
                    rewards=rewards,
                    dones=dones,
                    episode_infos=episode_info,
                )

            # Simulate learning
            collect_time = 0.05
            learn_time = 0.02

            # Simulate losses
            losses = {
                "policy": torch.tensor(0.5 - iteration * 0.04),
                "value": torch.tensor(0.3 - iteration * 0.02),
                "entropy": torch.tensor(0.1 + iteration * 0.001),
            }

            # Learning rate schedule
            learning_rate = 0.001 * (1 - iteration / total_iterations)

            # Action standard deviation
            action_std = torch.tensor([1.0 - iteration * 0.05])

            # Custom metrics
            custom_metrics = {
                "exploration_ratio": torch.tensor(0.9 - iteration * 0.05),
                "advantage_mean": torch.tensor(0.1 * iteration),
            }

            # Log training metrics
            logger.log(
                iteration=iteration,
                total_iterations=total_iterations,
                start_iteration=0,
                collect_time=collect_time,
                learn_time=learn_time,
                losses=losses,
                learning_rate=learning_rate,
                action_std=action_std,
                custom_metrics=custom_metrics,
                print_interval=2,  # Print every 2 iterations
            )

        # 3. Context Manager Usage
        loguru_logger.info("3. Context Manager Usage")
        loguru_logger.info("-" * 60)

        with Logger(
            LoggerConfig(log_dir=None, log_interval=1),  # No event file, just console
            num_envs=2,
            device="cpu",
        ) as ctx_logger:
            for i in range(3):
                rewards = torch.randn(2)
                dones = torch.zeros(2)
                ctx_logger.process_env_step(rewards, dones)
                loguru_logger.info(f"   Processed step {i + 1}/3")

        loguru_logger.info("   Context manager closed automatically")

        # 4. Episode Tracking
        loguru_logger.info("4. Episode Tracking")
        loguru_logger.info("-" * 60)

        logger2 = Logger(
            LoggerConfig(log_dir=None, log_interval=1),
            num_envs=3,
            device="cpu",
        )

        torch.manual_seed(42)
        for step in range(50):
            rewards = torch.randn(3) * 5 + 10  # Mean reward of 10
            # Episodes end randomly
            dones = (torch.rand(3) < 0.05).float()
            logger2.process_env_step(rewards, dones)

        if len(logger2.reward_buffer) > 0:
            mean_reward = sum(logger2.reward_buffer) / len(logger2.reward_buffer)
            loguru_logger.info(f"   Episodes completed: {len(logger2.reward_buffer)}")
            loguru_logger.info(f"   Mean reward: {mean_reward:.2f}")
        else:
            loguru_logger.info("   No episodes completed in this run")

        # 5. Checkpoints, which are CheckpointManager's job rather than the logger's
        loguru_logger.info("5. Checkpoint Saving")
        loguru_logger.info("-" * 60)

        model = torch.nn.Linear(10, 5)
        checkpoints = CheckpointManager(
            LoggerConfig(log_dir=log_dir, save_interval=5), run_dir=logger.log_dir
        )

        periodic = checkpoints.save({"model_state_dict": model.state_dict()}, iteration=5)
        best = checkpoints.save_best(
            {"model_state_dict": model.state_dict()}, metric=12.5, iteration=5
        )

        loguru_logger.info(f"   Periodic checkpoint: {periodic.name}")
        loguru_logger.info(f"   Best checkpoint:     {best.name}, kept while it is the best score")
        loguru_logger.info(f"   Due at iteration 5? {checkpoints.should_save(5)}")
        loguru_logger.info(f"   Due at iteration 6? {checkpoints.should_save(6)}")

        # 6. Metrics Buffers
        loguru_logger.info("6. Metrics Buffers")
        loguru_logger.info("-" * 60)

        loguru_logger.info(f"   Reward buffer size: {len(logger.reward_buffer)}")
        loguru_logger.info(f"   Episode length buffer size: {len(logger.episode_length_buffer)}")
        loguru_logger.info(f"   Total timesteps: {logger.total_timesteps}")
        loguru_logger.info(f"   Total time: {logger.total_time:.3f}s")

        # 7. Logging without writer
        loguru_logger.info("7. Logging without TensorBoard")
        loguru_logger.info("-" * 60)

        logger_no_tb = Logger(LoggerConfig(log_dir=None, log_interval=1), num_envs=2)
        logger_no_tb.log(
            iteration=0,
            total_iterations=1,
            losses={"policy": 0.5},
            custom_metrics={"ratio": 0.9},
        )
        loguru_logger.info("   Logged without TensorBoard writer (prints only)")

        logger.close()

    loguru_logger.info("=" * 60)
    loguru_logger.info("Logger example completed successfully!")


if __name__ == "__main__":
    main()
