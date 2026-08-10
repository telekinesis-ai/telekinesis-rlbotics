"""Example: MLPModel with policy and value networks.

This example demonstrates:
- Creating MLP-based policy and value networks
- Using distributions with MLPs
- Observation normalization
- Computing log probabilities and entropy
- Model exports (JIT, ONNX)
"""

import torch
from loguru import logger
from telekinesis.rlbotics.config import GaussianDistributionConfig, MLPConfig
from telekinesis.rlbotics.models import MLPModel


def main():
    """Demonstrate MLP model usage."""
    # Setup
    batch_size = 4
    obs_dim = 8
    act_dim = 3
    device = "cpu"

    logger.info("MLP Model Example")
    logger.info("=" * 60)

    # 1. Create policy network with distribution
    policy_cfg = MLPConfig(
        hidden_dims=(256, 256),
        activation="relu",
        obs_normalization=False,
        distribution_cfg=GaussianDistributionConfig(init_std=1.0, std_range=(1e-6, 1e6)),
    )
    policy = MLPModel(policy_cfg, input_dim=obs_dim, output_dim=act_dim)

    logger.info(f"1. Policy network created")
    logger.info(f"   Input dim: {obs_dim}, Output dim: {act_dim}")
    logger.info(f"   Hidden dims: (256, 256), Activation: relu")

    # 2. Create value network (critic)
    value_cfg = MLPConfig(hidden_dims=(256, 256), activation="relu")
    value = MLPModel(value_cfg, input_dim=obs_dim, output_dim=1)

    logger.info(f"2. Value network created")
    logger.info(f"   Input dim: {obs_dim}, Output dim: 1")

    # 3. Create dummy observations
    obs = torch.randn(batch_size, obs_dim, device=device)
    assert obs.shape == (batch_size, obs_dim)
    logger.info(f"3. Observations shape: {obs.shape}")

    # 4. Get deterministic actions (evaluation)
    with torch.no_grad():
        det_actions = policy(obs, stochastic=False)
    assert det_actions.shape == (batch_size, act_dim)
    logger.info(f"4. Deterministic actions shape: {det_actions.shape}")
    logger.info(f"   Sample: {det_actions[0]}")

    # 5. Get stochastic actions (exploration)
    with torch.no_grad():
        stoch_actions = policy(obs, stochastic=True)
    assert stoch_actions.shape == (batch_size, act_dim)
    logger.info(f"5. Stochastic actions shape: {stoch_actions.shape}")
    logger.info(f"   Sample: {stoch_actions[0]}")

    # 6. Compute log probabilities
    log_probs = policy.get_output_log_prob(stoch_actions)
    assert log_probs.shape == (batch_size,)
    assert torch.isfinite(log_probs).all()
    logger.info(f"6. Log probabilities shape: {log_probs.shape}")
    logger.info(f"   Values: {log_probs}")

    # 7. Compute policy entropy
    entropy = policy.output_entropy
    assert entropy.shape == (batch_size,)
    logger.info(f"7. Policy entropy shape: {entropy.shape}")
    logger.info(f"   Values: {entropy}")

    # 8. Get value estimates
    values = value(obs, stochastic=False).squeeze(-1)
    assert values.shape == (batch_size,)
    logger.info(f"8. Value estimates shape: {values.shape}")
    logger.info(f"   Values: {values}")

    # 9. Compute advantages (example)
    rewards = torch.randn(batch_size, device=device)
    with torch.no_grad():
        next_obs = torch.randn(batch_size, obs_dim, device=device)
        next_values = value(next_obs, stochastic=False)

    td_target = rewards + 0.99 * next_values
    advantages = td_target - values
    logger.info(f"9. Advantages shape: {advantages.shape}")
    logger.info(f"   Mean: {advantages.mean():.4f}, Std: {advantages.std():.4f}")

    # 10. Observation normalization
    policy_norm_cfg = MLPConfig(
        hidden_dims=(128, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg=GaussianDistributionConfig(),
    )
    policy_norm = MLPModel(policy_norm_cfg, input_dim=obs_dim, output_dim=act_dim)

    # Update normalization statistics
    obs_batch = torch.randn(1000, obs_dim) * 2.0 + 1.0
    policy_norm.update_normalization(obs_batch)
    logger.info(f"10. Observation normalization enabled")
    logger.info(f"    Updated with 1000 samples")

    # Normalized forward pass
    with torch.no_grad():
        normalized_actions = policy_norm(obs[:1], stochastic=False)
    logger.info(f"    Normalized action: {normalized_actions[0]}")

    # 11. Model exports
    jit_policy = policy.as_jit()
    onnx_policy = policy.as_onnx()
    logger.info(f"11. Model exports created")
    logger.info(f"    JIT model: {type(jit_policy).__name__}")
    logger.info(f"    ONNX model: {type(onnx_policy).__name__}")

    # 12. Verify export consistency
    with torch.no_grad():
        orig_out = policy(obs[:1], stochastic=False)
        jit_out = jit_policy(obs[:1])
        onnx_out = onnx_policy(obs[:1])

    assert torch.allclose(orig_out, jit_out, atol=1e-5)
    assert torch.allclose(orig_out, onnx_out, atol=1e-5)
    logger.info(f"    Export consistency verified ✓")

    # 13. Policy gradient computation (without no_grad)
    obs_grad = torch.randn(batch_size, obs_dim, device=device, requires_grad=True)
    actions_grad = policy(obs_grad, stochastic=True)
    log_probs_grad = policy.get_output_log_prob(actions_grad)
    loss = -log_probs_grad.mean()
    loss.backward()
    logger.info("13. Policy gradient computed")
    logger.info(f"    Gradient flow verified: {obs_grad.grad is not None} ✓")

    logger.info("=" * 60)
    logger.info("MLP model example completed successfully!")


if __name__ == "__main__":
    main()
