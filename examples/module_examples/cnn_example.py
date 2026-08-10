"""Example: CNNModel for vision-based policies.

This example demonstrates:
- Creating CNN-based policy networks for image inputs
- Combining CNN feature extraction with MLP head
- Using distributions with CNN models
- Computing actions from image observations
- Model exports for deployment
"""

import torch
from loguru import logger
from telekinesis.rlbotics.config import CNNConfig, CNNEncoderConfig, GaussianDistributionConfig
from telekinesis.rlbotics.models import CNNModel


def main():
    """Demonstrate CNN model usage."""
    # Setup
    batch_size = 4
    num_channels = 3  # RGB images
    img_height, img_width = 64, 64
    act_dim = 2  # 2D continuous action (e.g., forward/turn for a robot)
    device = "cpu"

    logger.info("CNN Model Example")
    logger.info("=" * 60)

    # 1. Create the convolutional encoder configuration
    encoder_cfg = CNNEncoderConfig(
        output_channels=(16, 32, 64),  # 3 conv layers
        kernel_size=3,
        stride=1,
        dilation=1,
        padding="none",       # No padding
        norm="none",          # No normalization
        activation="elu",
        max_pool=False,       # No pooling
        global_pool="none",   # No global pooling
        flatten=True,
    )

    logger.info(f"1. CNN encoder configuration")
    logger.info(f"   Input: {img_height}x{img_width}x{num_channels}")
    logger.info(f"   Channels: {encoder_cfg.output_channels}")
    logger.info(f"   Activation: {encoder_cfg.activation}, Flatten: {encoder_cfg.flatten}")

    # 2. Create policy network with CNN + squashed Gaussian
    policy_cfg = CNNConfig(
        hidden_dims=(256, 128),
        activation="relu",
        cnn_cfg=encoder_cfg,
        distribution_cfg=GaussianDistributionConfig(
            class_name="SquashedGaussianDistribution", init_std=0.5, std_range=(1e-6, 1e6)
        ),
    )

    policy = CNNModel(
        policy_cfg,
        input_dim=(img_height, img_width),
        input_channels=num_channels,
        output_dim=act_dim,
    )

    logger.info(f"2. CNN policy network created")
    logger.info(f"   Output dim: {act_dim}")
    logger.info(f"   MLP hidden dims: (256, 128)")

    # 3. Create dummy image observations (batch of RGB images)
    obs = torch.randn(batch_size, num_channels, img_height, img_width, device=device)
    assert obs.shape == (batch_size, num_channels, img_height, img_width)
    logger.info(f"3. Image observations shape: {obs.shape}")

    # 4. Get latent features from CNN
    with torch.no_grad():
        latent = policy.get_latent(obs)
    logger.info(f"4. CNN latent features shape: {latent.shape}")
    logger.info(f"   Latent dimension: {latent.shape[-1]}")

    # 5. Get deterministic actions (evaluation - no randomness)
    with torch.no_grad():
        det_actions = policy(obs, stochastic=False)
    assert det_actions.shape == (batch_size, act_dim)
    assert (det_actions >= -1.0).all() and (det_actions <= 1.0).all()
    logger.info(f"5. Deterministic actions shape: {det_actions.shape}")
    logger.info(f"   Range: [{det_actions.min():.4f}, {det_actions.max():.4f}]")
    logger.info(f"   Sample: {det_actions[0]}")

    # 6. Get stochastic actions (exploration - with noise)
    with torch.no_grad():
        stoch_actions = policy(obs, stochastic=True)
    assert stoch_actions.shape == (batch_size, act_dim)
    assert (stoch_actions >= -1.0).all() and (stoch_actions <= 1.0).all()
    logger.info(f"6. Stochastic actions shape: {stoch_actions.shape}")
    logger.info(f"   All in [-1, 1]: {(stoch_actions >= -1.0).all() and (stoch_actions <= 1.0).all()}")
    logger.info(f"   Sample: {stoch_actions[0]}")

    # 7. Compute log probabilities
    log_probs = policy.get_output_log_prob(stoch_actions)
    assert log_probs.shape == (batch_size,)
    assert torch.isfinite(log_probs).all()
    logger.info(f"7. Log probabilities shape: {log_probs.shape}")
    logger.info(f"   Values: {log_probs}")

    # 8. Compute policy entropy
    entropy = policy.output_entropy
    assert entropy.shape == (batch_size,)
    logger.info(f"8. Policy entropy shape: {entropy.shape}")
    logger.info(f"   Mean: {entropy.mean():.4f}")

    # 9. Policy parameters (mean and std)
    params = policy.output_distribution_params
    logger.info(f"9. Distribution parameters")
    logger.info(f"   Mean shape: {params[0].shape}")
    logger.info(f"   Std shape: {params[1].shape}")

    # 10. Different image resolutions
    logger.info(f"10. Testing different image resolutions")
    for img_size in [32, 48, 64, 128]:
        test_obs = torch.randn(2, num_channels, img_size, img_size, device=device)
        # The same config is reused; only the image shape passed at build time changes
        test_policy = CNNModel(
            policy_cfg,
            input_dim=(img_size, img_size),
            input_channels=num_channels,
            output_dim=act_dim,
        )
        with torch.no_grad():
            test_actions = test_policy(test_obs, stochastic=False)
        logger.info(f"    {img_size}x{img_size}: actions shape {test_actions.shape} ✓")

    # 11. Batch processing efficiency
    logger.info(f"11. Batch processing")
    with torch.no_grad():
        for batch in [1, 4, 16, 32]:
            batch_obs = torch.randn(batch, num_channels, img_height, img_width, device=device)
            batch_actions = policy(batch_obs, stochastic=False)
            assert batch_actions.shape == (batch, act_dim)
        logger.info(f"    Batch sizes [1, 4, 16, 32] processed successfully ✓")

    # 12. Model exports
    jit_policy = policy.as_jit()
    onnx_policy = policy.as_onnx()
    logger.info(f"12. Model exports created")
    logger.info(f"    JIT model: {type(jit_policy).__name__}")
    logger.info(f"    ONNX model: {type(onnx_policy).__name__}")

    # 13. Verify export consistency
    with torch.no_grad():
        orig_out = policy(obs, stochastic=False)
        jit_out = jit_policy(obs)
        onnx_out = onnx_policy(obs)

    assert torch.allclose(orig_out, jit_out, atol=1e-5)
    assert torch.allclose(orig_out, onnx_out, atol=1e-5)
    logger.info(f"    Export consistency verified ✓")

    # 14. Policy gradient through CNN (without no_grad)
    obs_grad = torch.randn(batch_size, num_channels, img_height, img_width, requires_grad=True)
    actions_grad = policy(obs_grad, stochastic=True)
    log_probs_grad = policy.get_output_log_prob(actions_grad)
    loss = -log_probs_grad.mean()
    loss.backward()
    logger.info("14. Policy gradient computation")
    logger.info(f"    Gradient flow verified through CNN: {obs_grad.grad is not None} ✓")

    logger.info("=" * 60)
    logger.info("CNN model example completed successfully!")


if __name__ == "__main__":
    main()
