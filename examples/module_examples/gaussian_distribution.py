"""Example: GaussianDistribution for unbounded continuous actions.

This example demonstrates using a Gaussian distribution with state-independent standard deviation
for policy parameterization. The Gaussian is suitable for unbounded action spaces (e.g., joint torques,
accelerations) where actions can be arbitrarily large.

Key concepts:
- Learnable log standard deviation (numerically stable)
- Sampling: draw from N(mean, std)
- Log probability: used for policy gradients (e.g., PPO)
- Entropy: encourages exploration during training
"""

import torch
import torch.nn as nn
from loguru import logger
from telekinesis.rlbotics.distributions import GaussianDistribution

def main():
    """Demonstrate GaussianDistribution usage."""
    # Setup
    batch_size = 4
    obs_dim = 8
    act_dim = 3
    device = "cpu"

    # Create distribution with learnable parameters
    dist = GaussianDistribution(
        act_dim=act_dim,
        init_std=1.0,
        std_range=(1e-6, 1e6),
    )

    logger.info("GaussianDistribution Example")
    logger.info("=" * 60)

    # 1. Policy network outputs actions (means)
    policy_network = nn.Linear(obs_dim, act_dim)
    policy_network.eval()

    # Dummy state observations
    observations = torch.randn(batch_size, obs_dim, device=device)
    assert observations.shape == (batch_size, obs_dim)
    logger.info(f"1. Observations shape: {observations.shape}")

    # Policy outputs action means
    action_means = policy_network(observations)
    assert action_means.shape == (batch_size, act_dim)
    logger.info(f"2. Action means shape: {action_means.shape} | sample: {action_means[0]}")

    # 2. Update distribution with means
    dist.update(action_means)
    logger.info(f"3. Distribution std: {dist.std[0]}")

    # 3. Sample actions (stochastic policy)
    actions = dist.sample()
    assert actions.shape == (batch_size, act_dim)
    logger.info(f"4. Sampled actions shape: {actions.shape} | sample: {actions[0]}")

    # 4. Compute log probabilities (needed for policy gradients)
    log_probs = dist.log_prob(actions)
    assert log_probs.shape == (batch_size,)
    logger.info(f"5. Log probabilities shape: {log_probs.shape} | values: {log_probs}")

    # 5. Compute entropy (encourages exploration)
    entropy = dist.entropy
    assert entropy.shape == (batch_size,)
    logger.info(f"6. Entropy shape: {entropy.shape} | values: {entropy}")

    # 6. Deterministic action (mean - for evaluation)
    deterministic_actions = dist.deterministic_output(action_means)
    assert deterministic_actions.shape == (batch_size, act_dim)
    logger.info(f"7. Deterministic actions: {deterministic_actions[0]}")

    # 7. KL divergence (important for trust region methods)
    old_action_means = torch.randn(batch_size, act_dim)
    dist_old = GaussianDistribution(act_dim=act_dim, init_std=1.0)
    dist_old.update(old_action_means)

    old_params = dist_old.params
    new_params = dist.params
    kl_div = dist.kl_divergence(old_params, new_params)
    assert kl_div.shape == (batch_size,)
    logger.info(f"8. KL divergence shape: {kl_div.shape} | KL(old || new): {kl_div}")


if __name__ == "__main__":
    main()
