"""Example: build a training configuration in Python, then train with it.

The counterpart to ``training_example.py``: the same run, with the configuration written as code
instead of read from YAML. Useful when a configuration is computed rather than fixed — a sweep, a
network sized from the observation, a hyperparameter that comes from somewhere else.

Four steps:

1. Build the configuration explicitly.
2. Create the environment and runner.
3. Train and export the policy.
4. Write the configuration out as YAML.

Run it:

    python examples/configuration_example.py

It prints where it wrote the configuration, which is a complete run that
``training_example.py`` can train from as-is:

    python examples/training_example.py <the file it wrote>

The configuration is a tree: an output distribution goes into the actor's model config, and the
model configs, the logger config and the algorithm config go into the runner config. Values are
validated as each piece is constructed, so a bad one is rejected here, not at the first step.

For a tour of the other config types, including the CNN ones for image observations, see
``module_examples/config_example.py``.

Requires Gymnasium:

    pip install "telekinesis-rlbotics[gym]"
"""

import argparse
import sys
from pathlib import Path

import yaml
from loguru import logger

from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
)
from telekinesis.rlbotics.envs.gym_env import GymnasiumVecEnv
from telekinesis.rlbotics.runner import OnPolicyRunner


def make_runner_config(
    obs_dim: int,
    log_dir: str,
    num_learning_iterations: int = 20,
    experiment: str = "configuration_example",
) -> OnPolicyRunnerConfig:
    """Build a runner configuration field by field.

    Args:
        obs_dim: Observation size of the task, used to size the networks. A 348-dimensional Humanoid
            observation needs more capacity than a 3-dimensional Pendulum one, which is the kind of
            decision a YAML file cannot make for itself.
        log_dir: Directory the run is written under.
        num_learning_iterations: Iterations to train for, which :meth:`OnPolicyRunner.learn` reads
            from the config when called with no argument.
        experiment: Experiment name, the directory its runs are grouped under.

    Returns:
        The runner configuration.
    """
    hidden_dims = (256, 256, 128) if obs_dim >= 100 else (64, 64)

    # What makes the actor stochastic. The std floor keeps exploration alive: without one the
    # standard deviation decays until the policy is near-deterministic and stops improving.
    distribution_cfg = GaussianDistributionConfig(
        class_name="GaussianDistribution",
        init_std=1.0,
        std_range=(0.2, 1.5),
    )

    # Observation normalization is worth keeping on for every task, since observation entries span
    # angles, velocities and contact forces with wildly different scales. It is exported with the
    # policy, so a deployment sees the same inputs training did.
    actor = MLPConfig(
        class_name="MLPModel",
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
        distribution_cfg=distribution_cfg,
    )

    # No distribution: a critic predicts one value rather than sampling
    critic = MLPConfig(
        class_name="MLPModel",
        hidden_dims=hidden_dims,
        activation="elu",
        obs_normalization=True,
    )

    algorithm = PPOConfig(
        # A registered name, or an import path such as "my_pkg.algorithms:MyPPO" for your own
        class_name="PPO",
        optimizer="adam",
        learning_rate=3.0e-4,
        num_learning_epochs=10,
        num_mini_batches=4,
        # An adaptive schedule holds each update inside the KL trust region, which is what stops the
        # reward climbing and then collapsing
        schedule="adaptive",
        desired_kl=0.01,
        clip_param=0.2,
        entropy_coef=0.01,
        gamma=0.99,
        lam=0.95,
        max_grad_norm=1.0,
        value_loss_coef=1.0,
    )

    # The run lands in <log_dir>/<experiment>/<timestamp>, with the event file, the config dump and
    # the checkpoints flat inside it. Resuming looks across the experiment's runs.
    logger_cfg = LoggerConfig(
        log_dir=log_dir,
        experiment=experiment,
        log_interval=10,
        save_interval=25,
        keep_last_n=5,
        save_best=True,
        log_video=False,
        resume=None,
    )

    return OnPolicyRunnerConfig(
        # Which observation groups each network reads. The Gymnasium adapter publishes one,
        # "observation"; a simulator with privileged state gives the critic its own group here.
        obs_groups={"actor": ["observation"], "critic": ["observation"]},
        num_learning_iterations=num_learning_iterations,
        num_steps_per_env=128,
        verbose=True,
        # A host sync every step, so leave it off unless an environment may return NaN
        check_for_nan=False,
        torch_compile_mode=None,
        logger=logger_cfg,
        algorithm=algorithm,
        actor=actor,
        critic=critic,
    )


def write_config(
    runner_cfg: OnPolicyRunnerConfig,
    path: Path,
    env_id: str,
    num_envs: int,
    device: str = "auto",
    framework: str = "gymnasium",
) -> Path:
    """Write a configuration out as YAML, in the shape the training examples read.

    ``to_dict`` is the inverse of the ``from_dict`` those examples use, so a configuration built
    here round-trips: the file it writes can be trained with directly. The ``env`` block beside it
    names the task, which the runner config has no field for.

    Args:
        runner_cfg: The configuration to write.
        path: File to write to.
        env_id: Gymnasium environment id the configuration was built for.
        num_envs: Environments to step in parallel.
        device: Device to simulate and train on.
        framework: Simulator the task belongs to, which is what training_example.py dispatches on.

    Returns:
        The path written.
    """
    data = {
        "env": {
            "framework": framework,
            "id": env_id,
            "num_envs": num_envs,
            "device": device,
        },
        "runner": runner_cfg.to_dict(),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False))

    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse. Defaults to None, which reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Build a training configuration in Python and train with it."
    )

    parser.add_argument(
        "-e",
        "--env-id",
        default="Pendulum-v1",
        help="Gymnasium environment ID.",
    )

    parser.add_argument(
        "-n",
        "--num-envs",
        type=int,
        default=8,
        help="Number of parallel environments.",
    )

    parser.add_argument(
        "-i",
        "--num-learning-iterations",
        type=int,
        default=20,
        help="Iterations to train for, written out as runner.num_learning_iterations. The default "
             "is a short demonstration run.",
    )

    parser.add_argument(
        "-d",
        "--device",
        default="auto",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device.",
    )

    parser.add_argument(
        "--log-dir",
        default=None,
        help="Log directory (default: <cwd>/logs).",
    )

    parser.add_argument(
        "--write-config",
        metavar="FILE",
        help="Where to write the configuration as YAML (default: beside the run).",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build a configuration, train with it, and write it out.

    Args:
        argv: Command line arguments. Defaults to None, which reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success.
    """
    args = parse_args(argv)
    log_dir = args.log_dir or str(Path.cwd() / "logs")

    # 1. Build the configuration explicitly. The environment comes first, so the networks can be
    # sized from the observation it reports.
    env = GymnasiumVecEnv(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=args.device,
    )

    logger.info(
        f"device={env.device}, "
        f"observations={env.obs_dim}, "
        f"actions={env.num_actions}, "
        f"episode_limit={env.max_episode_length}"
    )

    runner_cfg = make_runner_config(
        obs_dim=env.obs_dim,
        log_dir=log_dir,
        num_learning_iterations=args.num_learning_iterations,
    )

    logger.info(
        f"built a configuration: "
        f"networks={runner_cfg.actor.hidden_dims}, "
        f"learning_rate={runner_cfg.algorithm.learning_rate}, "
        f"steps_per_env={runner_cfg.num_steps_per_env}"
    )

    # 2. Create the runner
    runner = OnPolicyRunner(
        env=env,
        runner_cfg=runner_cfg,
        device=args.device,
    )

    # 3. Train and export the policy, for the configured number of iterations
    runner.learn()

    policy_path = runner.export()
    logger.info(f"exported policy: {policy_path}")

    env.close()

    # 4. Write the configuration out, so the same run is reproducible from YAML
    default_path = Path(policy_path).parent / f"{args.env_id}.generated.yaml"
    written = write_config(
        runner_cfg,
        Path(args.write_config) if args.write_config else default_path,
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=args.device,
    )
    logger.info(f"wrote configuration: {written}")
    logger.info(f"train with it: python examples/training_example.py {written}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
