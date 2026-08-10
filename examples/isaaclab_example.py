"""Example: train a PPO policy on an Isaac Lab task.

The same four steps as the other examples — read the inputs, build the runner, learn, deploy — but
on Isaac Lab, which runs Isaac Sim and so brings thousands of parallel environments, rendering and
a large task library::

    python examples/isaaclab_example.py --list-tasks              # what is registered
    python examples/isaaclab_example.py                           # Anymal C velocity tracking
    python examples/isaaclab_example.py -t Isaac-Cartpole-v0 -n 64 -i 50
    python examples/isaaclab_example.py --resume                  # continue from the last one
    python examples/isaaclab_example.py --resume best             # or the best scoring one
    python examples/isaaclab_example.py --record-video                # record an MP4 per checkpoint
    tensorboard --logdir logs                                     # watch the curves

See examples/isaaclab_play_example.py to run inference on an already-exported policy without
training, with an option to save a video.

Isaac Lab needs Python 3.11, Linux or Windows, and an NVIDIA GPU; there is no macOS build. Nothing
in this example has been run: it was written against Isaac Lab 2.3's source and is verified only
against a stubbed environment. Treat the hyperparameters as the usual starting point for legged
locomotion and watch ``Diagnostics/kl`` with ``Diagnostics/clip_fraction`` to see whether the
updates are sane.

Isaac Sim is launched by the adapter before the task is built, so do not import anything from
``isaaclab_tasks`` above.

Requires Isaac Lab:
    pip install "isaaclab[isaacsim,all]>=2.3" --extra-index-url https://pypi.nvidia.com
"""

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger

from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
)
from telekinesis.rlbotics.envs.base import observation_spec
from telekinesis.rlbotics.envs.isaaclab_env import (
    IsaacLabVecEnv,
    launch_simulator,
    registered_tasks,
    shutdown_simulator,
)
from telekinesis.rlbotics.policy import Policy
from telekinesis.rlbotics.runner import OnPolicyRunner

# Exit code that tells run_all_examples.py this example was skipped, not that it failed
SKIPPED = 2


def observation_groups(groups: list[str]) -> dict[str, list[str]]:
    """Map the task's observation groups onto the actor and the critic.

    Isaac Lab tasks publish a "policy" group, and those set up for an asymmetric actor-critic add a
    privileged "critic" group the critic alone reads.

    Args:
        groups: Observation groups the task publishes.

    Returns:
        The observation set the runner trains on.

    Raises:
        ValueError: If the task publishes no group the actor can read.
    """
    if "policy" not in groups:
        raise ValueError(
            f"This example expects the task to publish a 'policy' observation group, got {groups}."
        )
    return {"actor": ["policy"], "critic": ["critic"] if "critic" in groups else ["policy"]}


def make_runner_cfg(args: argparse.Namespace, groups: list[str]) -> OnPolicyRunnerConfig:
    """Build the training configuration from the command line.

    Args:
        args: Parsed arguments.
        groups: Observation groups the task publishes.

    Returns:
        The runner configuration.
    """
    return OnPolicyRunnerConfig(
        obs_groups=observation_groups(groups),
        num_steps_per_env=args.steps_per_env,
        verbose=True,
        # The run lands in <log_dir>/<experiment>/<timestamp>, with the event file, the config dump
        # and the checkpoints flat inside it. Resuming looks across the experiment's runs.
        logger=LoggerConfig(
            log_dir=args.log_dir,
            experiment=args.experiment_name,
            log_interval=10,
            save_interval=50,
            keep_last_n=5,
            resume=args.resume,
            log_video=args.record_video,
        ),
        algorithm=PPOConfig(
            learning_rate=args.learning_rate,
            num_learning_epochs=args.epochs,
            num_mini_batches=4,
            # An adaptive rate keeps each update inside the KL trust region, which is what stops the
            # reward climbing and then collapsing
            schedule="adaptive",
            desired_kl=0.01,
            clip_param=0.2,
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            lam=0.95,
            max_grad_norm=1.0,
        ),
        actor=MLPConfig(
            hidden_dims=args.hidden_dims,
            activation="elu",
            # The observation mixes joint positions, velocities, commands and contact state, whose
            # scales differ by orders of magnitude. The normalization is exported with the policy,
            # so deployment sees the same inputs training did
            obs_normalization=True,
            # Isaac Lab clips the actions itself, so the policy keeps a plain Gaussian and the std
            # floor is what keeps exploration alive
            distribution_cfg=GaussianDistributionConfig(init_std=1.0, std_range=(0.2, 2.0)),
        ),
        critic=MLPConfig(hidden_dims=args.hidden_dims, activation="elu", obs_normalization=True),
    )


def run_inference(policy_path: Path, env: IsaacLabVecEnv, num_steps: int = 10) -> None:
    """Act in the task with an exported policy.

    This is what deployment looks like: load the file, feed it observations, apply the actions. The
    policy itself needs neither torch nor the training stack and carries its own observation
    normalization; torch appears here only because an Isaac Lab environment speaks tensors, where a
    robot would hand over a numpy array.

    The loop runs inside ``torch.inference_mode()``, and has to. Training collects rollouts in that
    mode, so buffers the simulation allocates along the way are inference tensors from then on, and
    resetting the task outside the mode fails on their in-place update.

    Args:
        policy_path: Path to the exported policy.
        env: The task to act in.
        num_steps: Number of steps to run.
    """
    policy = Policy(policy_path)
    logger.info(f"loaded {policy!r}")

    with torch.inference_mode():
        obs = env.reset()
        for step in range(num_steps):
            action = policy.get_action(obs["policy"].cpu().numpy())
            obs, rewards, dones, _ = env.step(torch.as_tensor(action, device=env.device))
            logger.info(
                f"  step {step}: action in [{action.min():+.2f}, {action.max():+.2f}] "
                f"mean reward {rewards.mean():+.3f}, {int(dones.sum())} resets"
            )


def dims(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of layer widths.

    Args:
        text: Widths as text, such as "512,256,128".

    Returns:
        The widths.
    """
    return tuple(int(width) for width in text.split(","))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line.

    Args:
        argv: Arguments to parse. Defaults to None, which reads ``sys.argv``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        prog="isaaclab_example.py",
        description="Train a PPO policy on an Isaac Lab task.",
    )
    parser.add_argument(
        "-t", "--task", default="Isaac-Velocity-Flat-Anymal-C-v0", metavar="ID",
        help="registered Isaac Lab task id (default: %(default)s)",
    )
    parser.add_argument(
        "--list-tasks", action="store_true",
        help="print the registered task ids and exit",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=500,
        help="learning iterations (default: %(default)s; locomotion wants thousands)",
    )
    parser.add_argument(
        "-n", "--num-envs", type=int, default=4096,
        help="parallel environments (default: %(default)s)",
    )
    parser.add_argument(
        "-s", "--steps-per-env", type=int, default=24,
        help="steps per environment per iteration (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--device", default="auto", choices=["auto", "cpu", "cuda"],
        help="device to simulate and train on (default: %(default)s; needs an NVIDIA GPU)",
    )
    parser.add_argument(
        "--viewer", action="store_true",
        help="run Isaac Sim with its viewer window instead of headless",
    )
    parser.add_argument(
        "--clip-actions", type=float, default=None, metavar="LIMIT",
        help="symmetric limit Isaac Lab applies to the actions (default: no clipping)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3,
        help="initial learning rate, adapted to hold the KL target (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="optimization epochs per batch (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="discount factor (default: %(default)s)",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.005,
        help="entropy bonus (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-dims", type=dims, default=(512, 256, 128), metavar="N,N",
        help="layer widths of both networks (default: 512,256,128)",
    )
    parser.add_argument(
        "--log-dir", default=str(Path(__file__).parent.parent / "logs"), metavar="DIR",
        help="where logs and checkpoints go (default: <repo>/logs)",
    )
    parser.add_argument(
        "--experiment-name", default="isaaclab_ppo", metavar="NAME",
        help="experiment name, the directory its runs are grouped under (default: %(default)s)",
    )
    parser.add_argument(
        "--resume", nargs="?", const="last", default=None, metavar="WHICH",
        help="continue training: 'last', 'best', or a checkpoint path or file name "
             "(bare --resume means last)",
    )
    parser.add_argument(
        "--record-video", action="store_true",
        help="record an MP4 of the policy next to each checkpoint (needs "
        "'telekinesis-rlbotics[examples]', for imageio; rebuilds the Isaac Sim scene each time, "
        "so it is slow)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Train and deploy a policy.

    Args:
        argv: Command line arguments. Defaults to None, which reads ``sys.argv``.

    Returns:
        Process exit code: 0 on success, or SKIPPED when Isaac Lab is not installed.
    """
    # 1. Inputs. Building the environment launches Isaac Sim, which takes a while the first time
    args = parse_args(argv)
    try:
        if args.list_tasks:
            tasks = registered_tasks()
            logger.info(f"{len(tasks)} registered Isaac Lab tasks:")
            for task in tasks:
                logger.info(f"  {task}")
            shutdown_simulator()
            return 0

        # The simulator is a process-wide singleton: whichever call launches it first decides
        # whether cameras are enabled, so this has to happen before the training env is built if
        # a later recording env is going to need them
        launch_simulator(headless=not args.viewer, enable_cameras=args.record_video)

        env = IsaacLabVecEnv(
            args.task,
            num_envs=args.num_envs,
            device=args.device,
            clip_actions=args.clip_actions,
            headless=not args.viewer,
            render_mode="rgb_array" if args.record_video else None,
        )
    except ImportError as error:
        logger.warning(f"skipping: {error}")
        return SKIPPED

    spec = observation_spec(env)
    logger.info(
        f"{args.task} on {env.device}: {env.num_envs} envs, actions {env.num_actions}, "
        f"episode limit {env.max_episode_length}, observations {dict(spec)}"
    )

    # 2. Create the training runner
    runner_cfg = make_runner_cfg(args, list(spec))
    runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device=str(env.device))

    # 3. Learn, then export the policy for deployment
    runner.learn(num_learning_iterations=args.iterations)
    policy_path = runner.export()
    logger.info(f"exported {policy_path}")

    # 4. Deploy: load the exported policy and act with it
    run_inference(policy_path, env)

    # Isaac Sim does not exit with the task, so closing it is what ends the process cleanly
    env.close()
    shutdown_simulator()
    return 0


if __name__ == "__main__":
    sys.exit(main())
