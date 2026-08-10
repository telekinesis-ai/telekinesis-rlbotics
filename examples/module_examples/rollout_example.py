"""Example: Rollout storage for RL training.

This example demonstrates:
- Creating rollout storage for RL training
- Adding transitions during environment interaction
- Computing returns and advantages
- Generating mini-batches for training
- Generating batches for distillation
"""

import torch
from loguru import logger
from telekinesis.rlbotics.rollout import RolloutBuffer, Transition


def main():
    """Demonstrate rollout storage usage."""
    logger.info("Rollout Storage Example")
    logger.info("=" * 60)

    # Setup
    num_envs = 4
    num_transitions_per_env = 10
    obs_shape = (8,)  # Observation dimension
    actions_shape = (2,)  # Action dimension
    device = "cpu"

    # 1. Create RL Rollout Storage
    logger.info("1. Creating Rollout Storage for RL Training")
    logger.info("-" * 60)

    storage = RolloutBuffer(
        training_type="rl",
        num_envs=num_envs,
        num_transitions_per_env=num_transitions_per_env,
        obs_shape=obs_shape,
        actions_shape=actions_shape,
        device=device,
    )

    logger.info(f"   Training type: {storage.training_type}")
    logger.info(f"   Number of environments: {num_envs}")
    logger.info(f"   Transitions per environment: {num_transitions_per_env}")
    logger.info(f"   Observation shape: {obs_shape}")
    logger.info(f"   Action shape: {actions_shape}")
    logger.info(f"   Device: {device}")

    # 2. Simulate Rollout
    logger.info("2. Simulating Rollout Collection")
    logger.info("-" * 60)

    torch.manual_seed(42)
    for t in range(num_transitions_per_env):
        # Create a transition
        transition = Transition()
        transition.observations = torch.randn(num_envs, *obs_shape, device=device)
        transition.actions = torch.randn(num_envs, *actions_shape, device=device)
        transition.rewards = torch.randn(num_envs, 1, device=device)
        transition.dones = torch.randint(0, 2, (num_envs, 1), device=device).float()
        transition.values = torch.randn(num_envs, 1, device=device)
        transition.actions_log_prob = torch.randn(num_envs, 1, device=device)
        transition.distribution_params = (
            torch.randn(num_envs, *actions_shape, device=device),
            torch.randn(num_envs, *actions_shape, device=device),
        )

        # Add to storage
        storage.add_transition(transition)

        if (t + 1) % 3 == 0:
            logger.info(f"   Added {t + 1} transitions")

    logger.info(f"   Total transitions: {storage.step}")

    # 3. Compute Returns and Advantages
    logger.info("3. Computing Returns and Advantages (GAE)")
    logger.info("-" * 60)

    # Last values from the environment (bootstrap)
    last_values = torch.randn(num_envs, 1, device=device)

    storage.compute_returns_and_advantages(
        last_values=last_values, gamma=0.99, gae_lambda=0.95
    )

    logger.info(f"   Returns shape: {storage.returns.shape}")
    logger.info(f"   Advantages shape: {storage.advantages.shape}")
    logger.info(f"   Mean advantage: {storage.advantages.mean():.4f}")
    logger.info(f"   Std advantage: {storage.advantages.std():.4f}")
    logger.info(f"   Mean return: {storage.returns.mean():.4f}")

    # 4. Mini-batch Generator
    logger.info("4. Mini-batch Generator for Feedforward Networks")
    logger.info("-" * 60)

    num_mini_batches = 2
    num_epochs = 2
    batch_count = 0

    for batch in storage.mini_batch_generator(
        num_mini_batches=num_mini_batches, num_epochs=num_epochs
    ):
        batch_count += 1
        logger.info(
            f"   Batch {batch_count}: obs={batch.observations.shape}, "
            f"actions={batch.actions.shape}, adv={batch.advantages.shape}"
        )

        # Verify batch contents
        assert batch.observations is not None
        assert batch.actions is not None
        assert batch.values is not None
        assert batch.advantages is not None
        assert batch.returns is not None
        assert batch.old_actions_log_prob is not None
        assert batch.old_distribution_params is not None

        if batch_count >= 4:  # Show first 4 batches
            logger.info(f"   ... ({num_mini_batches * num_epochs} total batches)")
            break

    logger.info(f"   Total batches: {num_mini_batches * num_epochs}")
    logger.info(f"   Mini-batch size: {storage.num_envs * storage.num_transitions_per_env // num_mini_batches}")

    # 5. Distribution Parameters
    logger.info("5. Distribution Parameters")
    logger.info("-" * 60)

    logger.info(f"   Number of parameter tensors: {len(storage.distribution_params)}")
    for i, p in enumerate(storage.distribution_params):
        logger.info(f"   Param {i} shape: {p.shape}")

    # 6. Clear and Reuse Storage
    logger.info("6. Clearing Storage for Next Rollout")
    logger.info("-" * 60)

    logger.info(f"   Step before clear: {storage.step}")
    storage.clear()
    logger.info(f"   Step after clear: {storage.step}")

    # 7. Distillation Rollout Storage
    logger.info("7. Distillation Rollout Storage")
    logger.info("-" * 60)

    distill_storage = RolloutBuffer(
        training_type="distillation",
        num_envs=num_envs,
        num_transitions_per_env=5,
        obs_shape=obs_shape,
        actions_shape=actions_shape,
        device=device,
    )

    torch.manual_seed(42)
    for t in range(5):
        transition = Transition()
        transition.observations = torch.randn(num_envs, *obs_shape, device=device)
        transition.privileged_actions = torch.randn(
            num_envs, *actions_shape, device=device
        )
        transition.dones = torch.randint(0, 2, (num_envs, 1), device=device).float()

        distill_storage.add_transition(transition)

    logger.info(f"   Created distillation storage with {distill_storage.step} transitions")

    # 8. Distillation Generator
    logger.info("8. Distillation Generator")
    logger.info("-" * 60)

    batch_count = 0
    for batch in distill_storage.generator():
        batch_count += 1
        logger.info(
            f"   Batch {batch_count}: obs={batch.observations.shape}, "
            f"priv_actions={batch.privileged_actions.shape}"
        )

        if batch_count >= 3:
            logger.info(f"   ... ({distill_storage.num_transitions_per_env} total batches)")
            break

    # 9. Transition Reuse
    logger.info("9. Transition Reuse and Clearing")
    logger.info("-" * 60)

    transition = Transition()
    transition.observations = torch.randn(num_envs, *obs_shape, device=device)
    transition.actions = torch.randn(num_envs, *actions_shape, device=device)
    transition.rewards = torch.randn(num_envs, 1, device=device)
    transition.dones = torch.zeros(num_envs, 1, device=device)
    transition.values = torch.randn(num_envs, 1, device=device)
    transition.actions_log_prob = torch.randn(num_envs, 1, device=device)
    transition.distribution_params = (
        torch.randn(num_envs, *actions_shape, device=device),
        torch.randn(num_envs, *actions_shape, device=device),
    )

    logger.info(f"   Transition observations shape: {transition.observations.shape}")
    logger.info(f"   Transition actions shape: {transition.actions.shape}")

    transition.clear()
    logger.info(f"   After clear - observations: {transition.observations}")

    # 10. Storage Buffer Shapes
    logger.info("10. Storage Buffer Shapes")
    logger.info("-" * 60)

    logger.info(f"   Observations: {storage.observations.shape}")
    logger.info(f"   Actions: {storage.actions.shape}")
    logger.info(f"   Rewards: {storage.rewards.shape}")
    logger.info(f"   Dones: {storage.dones.shape}")
    logger.info(f"   Values: {storage.values.shape}")
    logger.info(f"   Actions log prob: {storage.actions_log_prob.shape}")
    logger.info(f"   Returns: {storage.returns.shape}")
    logger.info(f"   Advantages: {storage.advantages.shape}")

    logger.info("=" * 60)
    logger.info("Rollout storage example completed successfully!")


if __name__ == "__main__":
    main()
