"""Example: train and deploy a PPO policy on any supported simulator.

Loads a YAML configuration, trains, exports to ONNX, then runs the exported policy. The
configuration's ``env.framework`` decides which simulator is used, so one script covers all of them.

Requires the simulator's extra:

    pip install "telekinesis-rlbotics[gym]"        # gymnasium
    pip install "telekinesis-rlbotics[mjlab]"      # mjlab
    pip install "telekinesis-rlbotics[isaaclab]" --extra-index-url https://pypi.nvidia.com

To run, with the configuration path as the only argument:

    python examples/training_example.py configs/gymnasium/Humanoid-v5.yaml
    python examples/training_example.py configs/mjlab/Mjlab-Velocity-Flat-Unitree-G1.yaml
    python examples/training_example.py configs/isaaclab/Isaac-Velocity-Flat-Anymal-C-v0.yaml

Advanced usage — each option is named after the field it overrides, so ``--num-envs`` sets
``env.num_envs`` and ``-i`` sets ``runner.num_learning_iterations``:

    python examples/training_example.py configs/gymnasium/Pendulum-v1.yaml \\
        --num-envs 8 --num-learning-iterations 120 --device cpu

Try a config without writing into its own ``logs/`` directory:

    python examples/training_example.py configs/gymnasium/Pendulum-v1.yaml --log-dir /tmp/rlbotics

Resume from the newest checkpoint, or the best scoring one:

    python examples/training_example.py configs/gymnasium/Hopper-v5.yaml --resume
    python examples/training_example.py configs/gymnasium/Hopper-v5.yaml --resume best

Edit a configuration to change a run, one file per task:

    configs/<framework>/<task-id>.yaml

``env:`` names the simulator, the task, how many copies of it to step and what to step them on.
``runner:`` holds the networks, PPO hyperparameters, iteration count, logging, checkpoints and
videos. The framework and the task have no options, since they are what a configuration is for:
point at another file to train something else. To build the same configuration in Python, see
``configuration_example.py``.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from loguru import logger

from telekinesis.rlbotics.config import OnPolicyRunnerConfig
from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.policy import Policy
from telekinesis.rlbotics.runner import OnPolicyRunner

FRAMEWORKS = ("gymnasium", "mjlab", "isaaclab")

# Exit code that tells run_all_examples.py this example was skipped, not that it failed
SKIPPED = 2


def load_config(
    path: str | Path,
    args: argparse.Namespace,
) -> tuple[dict, OnPolicyRunnerConfig]:
    """Load a configuration, applying the command-line overrides.

    A file describes a whole run: ``env`` names the simulator, the task, how many copies of it to
    step and what to step them on, and ``runner`` maps onto :class:`OnPolicyRunnerConfig` field for
    field. So the configuration path is the only argument a run needs, and the options vary what
    changes between runs of the same task.

    Args:
        path: The configuration to read.
        args: Parsed arguments. An option left unset is None and leaves the file alone.

    Returns:
        The ``env`` settings, and the runner configuration.

    Raises:
        FileNotFoundError: If there is no file at ``path``.
        KeyError: If the configuration has no ``env`` or ``runner`` block.
        ValueError: If a value is invalid, raised by the config it belongs to.
    """
    data = yaml.safe_load(Path(path).read_text())
    env_data, runner_data = data["env"], data["runner"]

    # Applied to the data rather than to the built config, so the config classes still validate it.
    # The framework and the task have no options: they are what the configuration is for.
    if args.num_envs is not None:
        env_data["num_envs"] = args.num_envs

    if args.device is not None:
        env_data["device"] = args.device

    if args.num_learning_iterations is not None:
        runner_data["num_learning_iterations"] = args.num_learning_iterations

    if args.log_dir is not None:
        runner_data.setdefault("logger", {})["log_dir"] = args.log_dir

    if args.seed is not None:
        runner_data["seed"] = args.seed

    if args.resume is not None:
        runner_data.setdefault("logger", {})["resume"] = args.resume

    return env_data, OnPolicyRunnerConfig.from_dict(runner_data)


def make_env(env_cfg: dict, render: bool = False) -> VecEnv:
    """Build the vectorized environment the configuration asks for.

    Each adapter is imported inside its own branch, so only the simulator being used has to be
    installed. Beyond the shared keys, a framework reads what belongs to it: ``clip_actions`` for
    the two manager-based simulators, whose actions are joint targets rather than a normalized
    range, and ``headless`` for Isaac Sim.

    Args:
        env_cfg: The configuration's ``env`` block.
        render: Whether to build the environment so it can render, which recording a video needs.

    Returns:
        The environment, satisfying :class:`~telekinesis.rlbotics.envs.base.VecEnv`.

    Raises:
        ImportError: If the simulator is not installed, raised by its adapter.
        ValueError: If ``framework`` is not one of :data:`FRAMEWORKS`.
    """
    framework = env_cfg["framework"]
    shared = {
        "num_envs": env_cfg["num_envs"],
        "device": env_cfg.get("device", "auto"),
        "render_mode": "rgb_array" if render else None,
    }

    if framework == "gymnasium":
        from telekinesis.rlbotics.envs.gym_env import GymnasiumVecEnv

        return GymnasiumVecEnv(env_id=env_cfg["id"], **shared)

    if framework == "mjlab":
        from telekinesis.rlbotics.envs.mjlab_env import MjlabVecEnv

        return MjlabVecEnv(
            task=env_cfg["id"],
            clip_actions=env_cfg.get("clip_actions"),
            nconmax=env_cfg.get("nconmax"),
            **shared,
        )

    if framework == "isaaclab":
        from telekinesis.rlbotics.envs.isaaclab_env import IsaacLabVecEnv, launch_simulator

        # The simulator is a process-wide singleton: whichever call launches it first decides
        # whether cameras are enabled, so this has to happen before the training env is built if a
        # later env is ever going to need them too. Kept in sync with the headless/render_mode this
        # env is about to be built with, since a mismatch here silently wins and the correct values
        # passed below become a no-op.
        headless = env_cfg.get("headless", True)
        launch_simulator(headless=headless, enable_cameras=render)

        return IsaacLabVecEnv(
            task=env_cfg["id"],
            clip_actions=env_cfg.get("clip_actions"),
            headless=headless,
            **shared,
        )

    raise ValueError(f"Unknown env.framework '{framework}'. Expected one of {list(FRAMEWORKS)}.")


def run_inference(
    policy_path: Path,
    env: VecEnv,
    obs_groups: list[str],
    num_steps: int = 10,
) -> None:
    """Act in the environment with the exported policy.

    This is what deployment looks like: load the file, feed it observations, apply the actions. The
    policy needs neither torch nor the training stack — it carries its own observation
    normalization and action scaling, and takes numpy in — so torch appears here only because a
    vectorized environment speaks tensors.

    The loop runs under ``torch.inference_mode()``, which the GPU-resident simulators require:
    they allocate buffers during training that cannot be updated in place outside it.

    Args:
        policy_path: Path to the exported policy.
        env: The environment to act in.
        obs_groups: Groups the policy reads, matching the runner's ``obs_groups["actor"]``.
        num_steps: Number of steps to run.
    """
    policy = Policy(policy_path)
    logger.info(f"loaded {policy!r}")

    with torch.inference_mode():
        obs = env.reset()
        for step in range(num_steps):
            if len(obs_groups) == 1:
                actor_obs = obs[obs_groups[0]]
            else:
                actor_obs = torch.cat([obs[group] for group in obs_groups], dim=-1)

            action = policy.get_action(actor_obs.cpu().numpy())
            obs, rewards, dones, _ = env.step(torch.as_tensor(action, device=env.device))

            # Summarized rather than printed in full, since a task can have many actuators
            logger.info(
                f"step {step}: "
                f"action=[{action.min():+.2f}, {action.max():+.2f}], "
                f"reward={rewards.mean():+.3f}, "
                f"resets={int(dones.sum())}"
            )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Each option is named after the configuration field it overrides, and defaults to None so that
    an unset one leaves the file alone.

    Args:
        argv: Arguments to parse. Defaults to None, which reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and deploy a PPO policy on any supported simulator.",
        epilog="Each option is named after the configuration field it overrides.",
    )

    parser.add_argument(
        "config",
        help="Path to a YAML configuration, for example configs/gymnasium/Ant-v5.yaml.",
    )

    parser.add_argument(
        "-n",
        "--num-envs",
        type=int,
        help="Number of parallel environments. Overrides env.num_envs.",
    )

    parser.add_argument(
        "-d",
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        help="Training device. Overrides env.device.",
    )

    parser.add_argument(
        "-i",
        "--num-learning-iterations",
        type=int,
        help="Iterations to train for. Overrides runner.num_learning_iterations.",
    )

    parser.add_argument(
        "--log-dir",
        metavar="DIR",
        help="Where logs, checkpoints and videos go. Overrides runner.logger.log_dir. Handy for a "
             "quick test run without touching the config's own logs/ directory.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        help="Seed for model init, action sampling and mini-batch order. Overrides runner.seed.",
    )

    parser.add_argument(
        "--resume",
        nargs="?",
        const="last",
        default=None,
        metavar="WHICH",
        help="Continue from 'last', 'best', or a checkpoint path. Overrides runner.logger.resume.",
    )

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train and deploy a policy.

    Args:
        argv: Command line arguments. Defaults to None, which reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, 1 when the configuration could not be used, and
        SKIPPED when the simulator it asks for is not installed.
    """
    args = parse_args(argv)

    # 1. Load a YAML configuration
    try:
        env_cfg, runner_cfg = load_config(args.config, args)
    except (FileNotFoundError, KeyError, TypeError, ValueError) as error:
        logger.error(error)
        return 1

    logger.info(
        f"training {env_cfg['id']} on {env_cfg['framework']} with {args.config}: "
        f"{env_cfg['num_envs']} environments, {runner_cfg.num_learning_iterations} iterations"
    )

    # 2. Create the environment the configuration names, and the runner
    try:
        env = make_env(env_cfg, render=runner_cfg.logger.log_video)
    except ImportError as error:
        logger.warning(f"skipping: {error}")
        return SKIPPED
    except ValueError as error:
        logger.error(error)
        return 1

    logger.info(
        f"device={env.device}, "
        f"actions={env.num_actions}, "
        f"episode_limit={env.max_episode_length}"
    )

    # 3. Create the runner. Built directly rather than dispatched on runner.class_name, so a config
    # asking for a different runner is rejected here instead of silently training on this one.
    if runner_cfg.class_name not in ("OnPolicyRunner", OnPolicyRunner):
        logger.error(
            f"This example only builds OnPolicyRunner, got runner.class_name="
            f"{runner_cfg.class_name!r}. Use telekinesis.rlbotics.runner.create_runner() to build "
            "whatever class_name names."
        )
        return 1

    runner = OnPolicyRunner(
        env=env,
        runner_cfg=runner_cfg,
        device=str(env.device),
    )

    # 4. Train the policy
    runner.learn()

    # 5. Export the policy
    policy_path = runner.export()
    logger.info(f"exported policy: {policy_path}")

    # 6. Deploy the exported policy, before the environment is torn down
    run_inference(policy_path, env, runner_cfg.obs_groups["actor"])

    # 7. Close the environment
    env.close()
    if env_cfg["framework"] == "isaaclab":
        # Isaac Sim does not exit with the environment, and a script that leaves it running hangs
        from telekinesis.rlbotics.envs.isaaclab_env import shutdown_simulator

        shutdown_simulator()

    return 0


if __name__ == "__main__":
    sys.exit(main())
