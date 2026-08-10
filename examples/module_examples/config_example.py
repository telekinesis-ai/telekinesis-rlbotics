"""Example: building a training configuration in Python and passing it to the runner.

This example demonstrates every config, assembled from the inside out: the output distribution
goes into the actor model config, the model configs, the logger config and the algorithm config go
into the runner config, and that is all the runner needs to build and train.

Key concepts:
- GaussianDistributionConfig: output distribution of a stochastic policy
- MLPConfig: actor and critic models (a critic omits the distribution to get a deterministic value)
- CNNEncoderConfig and CNNConfig: a convolutional encoder plus an MLP head, for image inputs
- PPOConfig: algorithm hyperparameters
- LoggerConfig: where the run is written, how often it reports and checkpoints, and resuming
- OnPolicyRunnerConfig: the top-level bundle passed to the runner
- Validation: invalid values are rejected when the config is built, not at the first step
"""

import tempfile

import torch
from loguru import logger
from tensordict import TensorDict

from telekinesis.rlbotics.config import (
    CNNConfig,
    CNNEncoderConfig,
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
)
from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.runner import OnPolicyRunner


class SimpleEnv(VecEnv):
    """Minimal environment returning a policy observation group and a privileged one."""

    def __init__(self, num_envs: int = 4, obs_dim: int = 12, priv_dim: int = 6, num_actions: int = 3) -> None:
        """Initialize the environment."""
        # VecEnv is an interface, so the environment sets the attributes it promises
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = torch.device("cpu")
        self.obs_dim = obs_dim
        self.priv_dim = priv_dim
        self.obs = torch.randn(num_envs, obs_dim)
        self.priv = torch.randn(num_envs, priv_dim)

    def get_observations(self) -> TensorDict:
        """Return the current observations."""
        return TensorDict(
            {"policy": self.obs, "privileged": self.priv}, batch_size=(self.num_envs,), device=self.device
        )

    def reset(self) -> TensorDict:
        """Reset all environments."""
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments."""
        self.obs = self.obs + 0.01 * torch.randn_like(self.obs)
        rewards = -self.obs.norm(dim=1)
        dones = (torch.rand(self.num_envs) < 0.05).float()
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool)}
        return self.get_observations(), rewards, dones, extras


class VisionEnv(VecEnv):
    """Minimal environment returning a camera image group and a privileged state group."""

    def __init__(
        self,
        num_envs: int = 4,
        image_shape: tuple[int, int, int] = (3, 32, 32),
        priv_dim: int = 6,
        num_actions: int = 3,
    ) -> None:
        """Initialize the environment."""
        # VecEnv is an interface, so the environment sets the attributes it promises
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = torch.device("cpu")
        self.image_shape = image_shape
        self.priv_dim = priv_dim
        self.images = torch.rand(num_envs, *image_shape)
        self.priv = torch.randn(num_envs, priv_dim)

    def get_observations(self) -> TensorDict:
        """Return the current observations."""
        return TensorDict(
            {"camera": self.images, "privileged": self.priv}, batch_size=(self.num_envs,), device=self.device
        )

    def reset(self) -> TensorDict:
        """Reset all environments."""
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        """Execute one step in all environments."""
        self.images = (self.images + 0.01 * torch.randn_like(self.images)).clamp(0.0, 1.0)
        self.priv = self.priv + 0.01 * torch.randn_like(self.priv)
        rewards = -self.priv.norm(dim=1)
        dones = (torch.rand(self.num_envs) < 0.05).float()
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool)}
        return self.get_observations(), rewards, dones, extras


def mlp_config_example():
    """Build a full training config and train with it."""
    logger.info("MLPModel Configuration Example")
    logger.info("=" * 60)

    # 1. Config for the actor with GaussianDistribution
    dist_cfg = GaussianDistributionConfig(
        init_std=0.5,
        std_range=(0.05, 2.0),
        std_type="scalar",
        learn_std=True,
    )
    logger.info(f"1. Distribution config: {dist_cfg.to_dict()}")

    actor_cfg = MLPConfig(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=dist_cfg,
    )
    logger.info(f"2. Actor config: {actor_cfg.to_dict()}")

    # 2. Config for the critic
    critic_cfg = MLPConfig(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    logger.info(f"3. Critic config: {critic_cfg.to_dict()}")

    # 4. Config for PPO
    ppo_cfg = PPOConfig(
        learning_rate=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        schedule="adaptive",
        desired_kl=0.01,
        clip_param=0.2,
        entropy_coef=0.005,
        gamma=0.99,
        lam=0.95,
    )
    logger.info(f"5. PPO config: {ppo_cfg.to_dict()}")

    # 5. Config for the runner. LoggerConfig says where the run is written and how often it
    #    reports, how often it checkpoints and how many to keep.
    env = SimpleEnv()
    with tempfile.TemporaryDirectory() as log_dir:
        runner_cfg = OnPolicyRunnerConfig(
            obs_groups={"actor": ["policy"], "critic": ["policy", "privileged"]},
            num_steps_per_env=24,
            logger=LoggerConfig(
                log_dir=log_dir,
                experiment="config_example",
                log_interval=1,
                save_interval=10,
                keep_last_n=2,
            ),
            algorithm=ppo_cfg,
            actor=actor_cfg,
            critic=critic_cfg,
        )
        logger.info(f"6. Runner config: {runner_cfg.to_dict()}")

        # 6. Launch the runner
        runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device="cpu")

        logger.info(f"7. Actor input dim: {runner.alg.actor.input_dim} (policy only)")
        logger.info(f"8. Critic input dim: {runner.alg.critic.input_dim} (policy + privileged)")

        runner.learn(num_learning_iterations=2)
        logger.info(f"9. Trained {runner.current_learning_iteration + 1} iterations")

def cnn_config_example():
    """Build a training config for image observations and train with it."""
    logger.info("CNNModel Configuration Example")
    logger.info("=" * 60)

    # 1. Config for the actor with GaussianDistribution
    dist_cfg = GaussianDistributionConfig(
        init_std=0.5,
        std_range=(0.05, 2.0),
        std_type="scalar",
        learn_std=True,
    )
    logger.info(f"1. Distribution config: {dist_cfg.to_dict()}")

    # 2. Config for the convolutional encoder that consumes the camera images. Per-layer settings accept a scalar
    #    shared by all layers or one entry per layer.
    encoder_cfg = CNNEncoderConfig(
        output_channels=(16, 32),
        kernel_size=3,           # scalar: shared by both layers
        stride=(2, 1),           # sequence: one entry per layer
        padding="zeros",
        norm=("none", "layer"),
        activation="relu",
        max_pool=(False, True),
        global_pool="avg",
        flatten=True,
    )
    logger.info(f"2. CNN encoder config: {encoder_cfg.to_dict()}")

    # 3. Config for the actor: the encoder above, followed by an MLP head sized by hidden_dims
    actor_cfg = CNNConfig(
        hidden_dims=(256, 128),
        activation="elu",
        distribution_cfg=dist_cfg,
        cnn_cfg=encoder_cfg,
    )
    logger.info(f"3. Actor config: {actor_cfg.to_dict()}")

    # 4. Config for the critic. Value estimation does not need the camera, so it reads the privileged state directly
    #    with a plain MLP.
    critic_cfg = MLPConfig(
        hidden_dims=(256, 128),
        activation="elu",
        obs_normalization=True,
    )
    logger.info(f"4. Critic config: {critic_cfg.to_dict()}")

    # 5. Config for PPO
    ppo_cfg = PPOConfig(
        learning_rate=1e-3,
        num_learning_epochs=5,
        num_mini_batches=4,
        schedule="adaptive",
        desired_kl=0.01,
        clip_param=0.2,
        entropy_coef=0.005,
        gamma=0.99,
        lam=0.95,
    )
    logger.info(f"5. PPO config: {ppo_cfg.to_dict()}")

    # 6. Config for the runner. The actor sees the camera, the critic sees the privileged state.
    env = VisionEnv()
    with tempfile.TemporaryDirectory() as log_dir:
        runner_cfg = OnPolicyRunnerConfig(
            obs_groups={"actor": ["camera"], "critic": ["privileged"]},
            num_steps_per_env=8,
            logger=LoggerConfig(
                log_dir=log_dir,
                experiment="cnn_config_example",
                log_interval=1,
                save_interval=10,
                keep_last_n=2,
            ),
            algorithm=ppo_cfg,
            actor=actor_cfg,
            critic=critic_cfg,
        )
        logger.info(f"6. Runner config: {runner_cfg.to_dict()}")

        # 7. Launch the runner
        runner = OnPolicyRunner(env=env, runner_cfg=runner_cfg, device="cpu")

        logger.info(f"7. Actor encoder output dim: {runner.alg.actor.cnn.output_dim} (from the camera images)")
        logger.info(f"8. Critic input dim: {runner.alg.critic.input_dim} (privileged only)")

        runner.learn(num_learning_iterations=2)
        logger.info(f"9. Trained {runner.current_learning_iteration + 1} iterations")


if __name__ == "__main__":
    mlp_config_example()
    cnn_config_example()
