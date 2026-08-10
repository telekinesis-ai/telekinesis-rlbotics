"""Example: Normalization for observations and rewards.

This example demonstrates:
- Empirical normalization for observations
- Reward normalization with discounted variation
- Updating statistics and denormalization
- Practical use in training loops
"""

import torch
from loguru import logger
from telekinesis.rlbotics.normalization import (
    EmpiricalNormalization,
    EmpiricalDiscountedVariationNormalization,
)


def main():
    """Demonstrate normalization usage."""
    logger.info("Normalization Example")
    logger.info("=" * 60)

    # Setup
    obs_dim = 10
    reward_dim = 1
    batch_size = 32
    num_batches = 100
    device = "cpu"

    # 1. Observation Normalization
    logger.info("1. Empirical Observation Normalization")
    logger.info("-" * 60)

    obs_norm = EmpiricalNormalization(shape=obs_dim, eps=1e-2)
    obs_norm.train()  # Set to training mode to enable updates

    # Generate dummy observations (e.g., from multiple episodes)
    observations = []
    for batch_idx in range(num_batches):
        # Simulate observations with different statistics in each batch
        batch_obs = torch.randn(batch_size, obs_dim)
        batch_obs = batch_obs * (1 + batch_idx * 0.01) + batch_idx * 0.05
        observations.append(batch_obs)

    # Update normalization statistics
    for batch_idx, batch_obs in enumerate(observations[:10]):
        obs_norm.update(batch_obs)
        if batch_idx % 3 == 0:
            logger.info(
                f"   After batch {batch_idx}: mean={obs_norm.mean[:3].tolist()}, "
                f"std={obs_norm.std[:3].tolist()}"
            )

    logger.info(f"   Total updates: {obs_norm.count}")
    logger.info(f"   Final mean: {obs_norm.mean[:5].tolist()}")
    logger.info(f"   Final std: {obs_norm.std[:5].tolist()}")

    # 2. Normalization in Action
    logger.info("2. Normalizing and Denormalizing Observations")
    logger.info("-" * 60)

    obs_sample = observations[-1][:4]  # Take 4 samples
    logger.info(f"   Original obs (first 4, first 3 dims):")
    logger.info(f"   {obs_sample[:, :3].tolist()}")

    # Normalize observations
    obs_norm.eval()  # Set to eval mode (no updates)
    normalized_obs = obs_norm(obs_sample)
    logger.info(f"   Normalized obs (first 4, first 3 dims):")
    logger.info(f"   {normalized_obs[:, :3].tolist()}")

    # Denormalize observations
    denormalized_obs = obs_norm.inverse(normalized_obs)
    logger.info(f"   Denormalized obs (should match original):")
    logger.info(f"   {denormalized_obs[:, :3].tolist()}")
    logger.info(f"   Reconstruction error: {torch.mean(torch.abs(obs_sample - denormalized_obs)):.6f}")

    # 3. Properties Access
    logger.info("3. Accessing Normalization Statistics")
    logger.info("-" * 60)

    logger.info(f"   Mean tensor shape: {obs_norm.mean.shape}")
    logger.info(f"   Std tensor shape: {obs_norm.std.shape}")
    logger.info(f"   Count: {obs_norm.count.item()}")
    logger.info(f"   Epsilon: {obs_norm.eps}")

    # 4. Reward Normalization with Discounted Variation
    logger.info("4. Reward Normalization (Discounted Variation)")
    logger.info("-" * 60)

    reward_norm = EmpiricalDiscountedVariationNormalization(
        shape=reward_dim,
        eps=1e-2,
        gamma=0.99,
    )
    reward_norm.train()

    # Simulate rewards from an episode
    episode_rewards = []
    for step in range(50):
        # Simulate rewards that grow over the episode
        reward = torch.tensor([[0.1 + step * 0.01]], dtype=torch.float32)
        episode_rewards.append(reward)

    logger.info(f"   Simulating 50 episode steps...")
    for step, reward in enumerate(episode_rewards[:10]):
        normalized_reward = reward_norm(reward)
        if step % 3 == 0:
            logger.info(
                f"   Step {step}: raw={reward.item():.4f}, normalized={normalized_reward.item():.4f}"
            )

    logger.info(f"   Reward normalization std: {reward_norm.emp_norm.std.item():.6f}")

    # 5. Batch Processing
    logger.info("5. Batch Processing with Normalization")
    logger.info("-" * 60)

    obs_norm_batch = EmpiricalNormalization(shape=obs_dim, eps=1e-2)
    obs_norm_batch.train()

    for batch_idx in range(5):
        batch = torch.randn(batch_size, obs_dim) + batch_idx * 0.1
        obs_norm_batch.update(batch)

    batch_test = torch.randn(batch_size, obs_dim)
    obs_norm_batch.eval()
    normalized_batch = obs_norm_batch(batch_test)

    logger.info(f"   Original batch mean: {batch_test.mean(dim=0)[:5].tolist()}")
    logger.info(f"   Original batch std: {batch_test.std(dim=0)[:5].tolist()}")
    logger.info(f"   Normalized batch mean: {normalized_batch.mean(dim=0)[:5].tolist()}")
    logger.info(f"   Normalized batch std: {normalized_batch.std(dim=0)[:5].tolist()}")

    # 6. Training vs Evaluation Mode
    logger.info("6. Training vs Evaluation Mode Behavior")
    logger.info("-" * 60)

    obs_norm_mode = EmpiricalNormalization(shape=5, eps=1e-2)

    # Training mode - updates statistics
    obs_norm_mode.train()
    batch1 = torch.randn(10, 5)
    obs_norm_mode.update(batch1)
    count_after_update = obs_norm_mode.count.item()
    logger.info(f"   Training mode - after update: count={count_after_update}")

    # Evaluation mode - no updates
    obs_norm_mode.eval()
    batch2 = torch.randn(10, 5)
    obs_norm_mode.update(batch2)  # This call is ignored in eval mode
    count_no_change = obs_norm_mode.count.item()
    logger.info(f"   Evaluation mode - after update: count={count_no_change}")
    logger.info(f"   Count unchanged: {count_after_update == count_no_change}")

    # 7. Until Parameter (Learning Cutoff)
    logger.info("7. Learning Cutoff with 'until' Parameter")
    logger.info("-" * 60)

    obs_norm_until = EmpiricalNormalization(shape=5, eps=1e-2, until=100)
    obs_norm_until.train()

    for i in range(5):
        batch = torch.randn(30, 5)
        obs_norm_until.update(batch)
        logger.info(f"   Batch {i}: count={obs_norm_until.count.item()}, continues={obs_norm_until.count < 100}")

    logger.info(f"   Learning stopped when count exceeded 'until'={obs_norm_until.until}")

    # 8. Multi-dimensional Shapes
    logger.info("8. Multi-dimensional Shape Support")
    logger.info("-" * 60)

    # Image-like observations (C, H, W)
    image_shape = (3, 32, 32)
    img_norm = EmpiricalNormalization(shape=image_shape, eps=1e-2)
    img_norm.train()

    batch_images = torch.randn(16, *image_shape)
    img_norm.update(batch_images)
    normalized_images = img_norm(batch_images)

    logger.info(f"   Image shape: {image_shape}")
    logger.info(f"   Batch images shape: {batch_images.shape}")
    logger.info(f"   Normalized images shape: {normalized_images.shape}")
    logger.info(f"   Normalization mean shape: {img_norm.mean.shape}")
    logger.info(f"   Normalization std shape: {img_norm.std.shape}")

    logger.info("=" * 60)
    logger.info("Normalization example completed successfully!")


if __name__ == "__main__":
    main()
