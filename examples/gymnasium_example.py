"""Example: train a PPO policy on any Gymnasium continuous control task.

Four steps: read the inputs, build the runner, learn, deploy. Everything the environment decides —
observation size, number of actuators, action bounds, episode limit — is read from the environment
id, so switching task means switching one argument::

    python examples/gymnasium_example.py --help
    python examples/gymnasium_example.py                 # Humanoid-v5, roughly 20 minutes
    python examples/gymnasium_example.py -e Hopper-v5    # any continuous control id works
    python examples/gymnasium_example.py --resume        # continue from the last checkpoint
    python examples/gymnasium_example.py --resume best   # or from the best scoring one
    python examples/gymnasium_example.py --record-video     # record an MP4 next to each checkpoint
    tensorboard --logdir logs                            # watch the curves

To look at a policy after a run, see examples/module_examples/video_example.py, which loads a
checkpoint and records it.

The default task is Humanoid-v5: 17 actuators and a 348-dimensional observation, rewarded for
forward velocity and for staying alive, so the policy has to learn to stand, then walk, then run.
Measured at the defaults, mean reward goes from 60 to 622 and mean episode length from 13 to 129
steps over 600 iterations. A gait that reads as running needs millions of steps, so treat the
default 3000 iterations as a start rather than a finish. Watch Episodes/mean_length first, since
staying upright is what the return is built on, and Diagnostics/kl with Diagnostics/clip_fraction to
see whether the updates are sane.

The defaults suit a hard, many-actuator task. Smaller tasks take a larger learning rate, more epochs
over each batch and more entropy, and they get there in minutes::

    -e Pendulum-v1 -n 8 -s 128 -i 120 --learning-rate 3e-4 --epochs 10 --entropy-coef 0.01
    -e MountainCarContinuous-v0 -n 8 -s 128 -i 300 --learning-rate 3e-4 --epochs 10
    -e Reacher-v5 -n 8 -s 128 -i 500 --learning-rate 3e-4 --epochs 10 --entropy-coef 0.001
    -e Hopper-v5 -n 32 -s 64 -i 1500 --epochs 5 --entropy-coef 0.001

The Pendulum recipe is the quick check that everything works: measured, its mean reward goes from
about -1080 to -211 over 120 iterations, which is a solved swing-up, in about two minutes.

Discrete tasks (CartPole-v1, Acrobot-v1, MountainCar-v0) are rejected with a clear error: PPO here
uses a Gaussian policy, which needs a continuous action space.

Requires Gymnasium: pip install "telekinesis-rlbotics[gym]"
"""

import argparse
from pathlib import Path

import gymnasium as gym
from loguru import logger

from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
)
from telekinesis.rlbotics.envs.gym_env import GymnasiumVecEnv
from telekinesis.rlbotics.policy import Policy
from telekinesis.rlbotics.runner import OnPolicyRunner


def make_runner_cfg(args: argparse.Namespace, obs_dim: int) -> OnPolicyRunnerConfig:
    """Build the training configuration from the command line.

    Args:
        args: Parsed arguments.
        obs_dim: Observation size of the task, used to size the networks when ``--hidden-dims`` is
            not given. A 348-dimensional Humanoid observation needs more capacity than a
            3-dimensional Pendulum one.

    Returns:
        The runner configuration.
    """
    hidden_dims = args.hidden_dims or ((256, 256, 128) if obs_dim >= 100 else (64, 64))

    return OnPolicyRunnerConfig(
        obs_groups={"actor": ["observation"], "critic": ["observation"]},
        num_steps_per_env=args.steps_per_env,
        verbose=True,
        # The run lands in <log_dir>/<experiment>/<timestamp>, with the event file, the config dump
        # and the checkpoints flat inside it. Resuming looks across the experiment's runs.
        logger=LoggerConfig(
            log_dir=args.log_dir,
            experiment=args.experiment_name,
            log_interval=10,
            save_interval=100,
            keep_last_n=5,
            resume=args.resume,
            log_video=args.record_video,
        ),
        algorithm=PPOConfig(
            learning_rate=args.learning_rate,
            num_learning_epochs=args.epochs,
            num_mini_batches=4,
            # An adaptive rate keeps each update inside the KL trust region, which is what stops the
            # reward climbing and then collapsing. Measured on Humanoid-v5 over 250-400 iterations:
            # a fixed 1e-4 took updates of KL 0.137 with clip fraction 0.52, ~14x the target, while
            # adaptive settled at KL 0.016 and clip fraction 0.18 at the same reward.
            schedule="adaptive",
            desired_kl=0.01,
            clip_param=0.2,
            # The entropy bonus has no target, it just pushes the standard deviation up until the
            # policy gradient balances it, so watch Policy/action_std when changing it.
            entropy_coef=args.entropy_coef,
            gamma=args.gamma,
            lam=0.95,
            max_grad_norm=1.0,
        ),
        actor=MLPConfig(
            hidden_dims=hidden_dims,
            activation="elu",
            # Worth keeping on for every task: observation entries span angles, velocities and
            # contact forces with wildly different scales, and the normalization is exported with
            # the policy so deployment sees the same inputs training did.
            obs_normalization=True,
            # A plain Gaussian, with the action scaling applied by the environment wrapper, as
            # standard PPO on MuJoCo does. The squashed variant is numerically fragile at 17 action
            # dimensions: once the mean drifts, tanh saturates, its log probability explodes and the
            # ratio overflows.
            #
            # The std floor keeps exploration alive. Without one the standard deviation decays until
            # the policy is near-deterministic and stops improving: on Humanoid-v5 over 6000
            # iterations it went 1.00 -> 0.29 while the reward peaked at 1788 and fell back to 611.
            distribution_cfg=GaussianDistributionConfig(init_std=1.0, std_range=(0.2, 1.5)),
        ),
        critic=MLPConfig(hidden_dims=hidden_dims, activation="elu", obs_normalization=True),
    )


def run_inference(policy_path: Path, env_id: str, num_steps: int = 10) -> None:
    """Act in the environment with an exported policy.

    This is what deployment looks like: load the file, feed it observations, apply the actions. It
    needs neither torch nor the training stack, and the policy carries its own observation
    normalization and action scaling.

    Args:
        policy_path: Path to the exported policy.
        env_id: Gymnasium environment id.
        num_steps: Number of steps to run.
    """
    policy = Policy(policy_path)
    logger.info(f"loaded {policy!r}")

    env = gym.make(env_id)
    obs, _ = env.reset(seed=0)

    total = 0.0
    for step in range(num_steps):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        # Summarized rather than printed in full, since a task can have many actuators
        logger.info(
            f"  step {step}: action in [{action.min():+.2f}, {action.max():+.2f}] "
            f"reward {reward:+.3f}"
        )
        if terminated or truncated:
            obs, _ = env.reset()
    env.close()

    logger.info(f"return over {num_steps} steps: {total:.1f}")


def dims(text: str) -> tuple[int, ...]:
    """Parse a comma-separated list of layer widths.

    Args:
        text: Widths as text, such as "256,256,128".

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
        prog="gymnasium_example.py",
        description="Train a PPO policy on a Gymnasium continuous control task.",
    )
    parser.add_argument(
        "-e", "--env-id", default="Humanoid-v5", metavar="ID",
        help="environment id, needs a continuous action space (default: %(default)s)",
    )
    parser.add_argument(
        "-i", "--iterations", type=int, default=3000,
        help="learning iterations (default: %(default)s, ~3M steps and roughly 20 minutes)",
    )
    parser.add_argument(
        "-n", "--num-envs", type=int, default=32,
        help="parallel environments (default: %(default)s)",
    )
    parser.add_argument(
        "-s", "--steps-per-env", type=int, default=32,
        help="steps per environment per iteration (default: %(default)s)",
    )
    parser.add_argument(
        "-d", "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"],
        help="device to train on (default: %(default)s; cpu is usually faster at these sizes)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="initial learning rate, adapted to hold the KL target (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs", type=int, default=2,
        help="optimization epochs per batch; raise it for easier tasks (default: %(default)s)",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99,
        help="discount factor (default: %(default)s)",
    )
    parser.add_argument(
        "--entropy-coef", type=float, default=0.0011,
        help="entropy bonus; ~0.01 for few-actuator tasks (default: %(default)s)",
    )
    parser.add_argument(
        "--hidden-dims", type=dims, default=None, metavar="N,N",
        help="layer widths of both networks (default: from the observation size)",
    )
    parser.add_argument(
        "--log-dir", default=str(Path(__file__).parent.parent / "logs"), metavar="DIR",
        help="where logs and checkpoints go (default: <repo>/logs)",
    )
    parser.add_argument(
        "--experiment-name", default="gymnasium_ppo", metavar="NAME",
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
        "'telekinesis-rlbotics[examples]', for imageio)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Train and deploy a policy.

    Args:
        argv: Command line arguments. Defaults to None, which reads ``sys.argv``.
    """
    # 1. Inputs
    args = parse_args(argv)
    env = GymnasiumVecEnv(
        args.env_id,
        args.num_envs,
        args.device,
        render_mode="rgb_array" if args.record_video else None,
    )
    bounds = (
        f"[{env.action_low.min():+.2f}, {env.action_high.max():+.2f}]"
        if env.action_low is not None
        else "unbounded"
    )
    logger.info(
        f"{args.env_id} on {env.device}: {env.num_envs} envs, obs {env.obs_dim}, "
        f"actions {env.num_actions} in {bounds}, episode limit {env.max_episode_length}"
    )

    # 2. Create the training runner
    runner_cfg = make_runner_cfg(args, env.obs_dim)
    runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device=args.device)

    # 3. Learn, then export the policy for deployment
    runner.learn(num_learning_iterations=args.iterations)
    policy_path = runner.export()
    logger.info(f"exported {policy_path}")
    env.close()

    # 4. Deploy: load the exported policy and act with it
    run_inference(policy_path, args.env_id)


if __name__ == "__main__":
    main()
