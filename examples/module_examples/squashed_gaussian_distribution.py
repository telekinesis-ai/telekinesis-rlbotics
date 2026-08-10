"""Example: SquashedGaussianDistribution for bounded continuous control.

This example demonstrates using a tanh-squashed Gaussian distribution for bounded action spaces.
Actions are sampled from a Gaussian, then passed through tanh to produce bounded outputs in [-1, 1].

The log probability correctly accounts for the Jacobian of the tanh transformation, which is
critical for unbiased policy gradient estimates.

Key concepts:
- Sampling: z ~ N(mean, std), then y = tanh(z) ∈ [-1, 1]
- Log probability includes Jacobian: log p(y) = log p(z) - Σ log(1 - tanh(z)²)
- Standard for modern RL algorithms (SAC, PPO, TRPO)
- Better numerical stability at action boundaries than clipping
"""

import torch
from torch import nn
from loguru import logger
from telekinesis.rlbotics.distributions import SquashedGaussianDistribution


def main():
    """Demonstrate SquashedGaussianDistribution usage."""
    # Setup
    batch_size = 4
    obs_dim = 16
    act_dim = 3
    device = "cpu"

    # Create squashed Gaussian distribution
    dist = SquashedGaussianDistribution(
        act_dim=act_dim,
        init_std=0.5,
        std_range=(1e-6, 1e6),
    )

    logger.info("SquashedGaussianDistribution Example")
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

    # 2. Update distribution
    dist.update(action_means)
    logger.info(f"3. Distribution std: {dist.std[0]}")

    # 3. Sample actions with tanh squashing
    sampled_actions = dist.sample()
    assert sampled_actions.shape == (batch_size, act_dim)
    assert (sampled_actions >= -1.0).all() and (sampled_actions <= 1.0).all()
    min_val = sampled_actions.min().item()
    max_val = sampled_actions.max().item()
    logger.info(
        f"4. Sampled actions (squashed) shape: {sampled_actions.shape} | "
        f"range: [{min_val:.4f}, {max_val:.4f}]"
    )

    # 4. Log probability with Jacobian correction
    log_probs = dist.log_prob(sampled_actions)
    assert log_probs.shape == (batch_size,)
    assert torch.isfinite(log_probs).all()
    logger.info(f"5. Log probabilities shape: {log_probs.shape} | values: {log_probs}")

    # 5. Entropy (approximate)
    entropy = dist.entropy
    assert entropy.shape == (batch_size,)
    logger.info(f"6. Entropy shape: {entropy.shape} | values: {entropy}")

    # 6. Deterministic action (mean)
    deterministic_actions = dist.deterministic_output(action_means)
    assert deterministic_actions.shape == (batch_size, act_dim)
    logger.info(f"7. Deterministic actions: {deterministic_actions[0]}")

    # 7. Test boundary behavior
    boundary_actions = torch.tensor([
        [0.9999, -0.9999, 0.5],
        [0.999, -0.999, 0.0],
        [0.99, -0.99, -0.5],
        [0.95, -0.95, 0.0],
    ], device=device)

    boundary_means = policy_network(torch.randn(batch_size, obs_dim))
    dist_boundary = SquashedGaussianDistribution(act_dim=act_dim, init_std=0.5)
    dist_boundary.update(boundary_means)
    boundary_log_probs = dist_boundary.log_prob(boundary_actions)
    assert boundary_log_probs.shape == (batch_size,)
    assert torch.isfinite(boundary_log_probs).all()
    logger.info(f"8. Boundary log probs shape: {boundary_log_probs.shape} | all finite: {torch.isfinite(boundary_log_probs).all()}")


if __name__ == "__main__":
    main()
