"""Tests for rollout storage."""

import pytest
import torch
from telekinesis.rlbotics.rollout import RolloutBuffer, RolloutBufferBatch, Transition


class TestTransition:
    """Test Transition class."""

    def test_initialization(self):
        """Test transition initialization."""
        transition = Transition()

        assert transition.observations is None
        assert transition.actions is None
        assert transition.rewards is None
        assert transition.dones is None
        assert transition.values is None
        assert transition.actions_log_prob is None
        assert transition.distribution_params is None
        assert transition.privileged_actions is None
        assert transition.hidden_states == (None, None)

    def test_clear(self):
        """Test clearing a transition."""
        transition = Transition()
        transition.observations = torch.randn(4, 8)
        transition.actions = torch.randn(4, 2)

        transition.clear()

        assert transition.observations is None
        assert transition.actions is None


class TestBatch:
    """Test RolloutBufferBatch class."""

    def test_initialization_empty(self):
        """Test batch initialization without arguments."""
        batch = RolloutBufferBatch()

        assert batch.observations is None
        assert batch.actions is None
        assert batch.values is None
        assert batch.advantages is None
        assert batch.returns is None
        assert batch.old_actions_log_prob is None
        assert batch.old_distribution_params is None
        assert batch.privileged_actions is None
        assert batch.dones is None
        assert batch.hidden_states == (None, None)
        assert batch.masks is None

    def test_initialization_with_data(self):
        """Test batch initialization with data."""
        obs = torch.randn(32, 8)
        actions = torch.randn(32, 2)
        values = torch.randn(32, 1)

        batch = RolloutBufferBatch(
            observations=obs,
            actions=actions,
            values=values,
        )

        assert torch.equal(batch.observations, obs)
        assert torch.equal(batch.actions, actions)
        assert torch.equal(batch.values, values)


class TestRolloutBufferRL:
    """Test RolloutBuffer for RL training."""

    def test_initialization_rl(self):
        """Test initialization for RL training."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        assert storage.training_type == "rl"
        assert storage.num_envs == 4
        assert storage.num_transitions_per_env == 10
        assert storage.observations.shape == (10, 4, 8)
        assert storage.actions.shape == (10, 4, 2)
        assert storage.rewards.shape == (10, 4, 1)
        assert storage.dones.shape == (10, 4, 1)
        assert storage.values.shape == (10, 4, 1)
        assert storage.actions_log_prob.shape == (10, 4, 1)
        assert storage.returns.shape == (10, 4, 1)
        assert storage.advantages.shape == (10, 4, 1)
        assert storage.step == 0

    def test_add_transition(self):
        """Test adding a transition."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        transition = Transition()
        transition.observations = torch.randn(4, 8)
        transition.actions = torch.randn(4, 2)
        transition.rewards = torch.randn(4, 1)
        transition.dones = torch.zeros(4, 1)
        transition.values = torch.randn(4, 1)
        transition.actions_log_prob = torch.randn(4, 1)
        transition.distribution_params = (
            torch.randn(4, 2),
            torch.randn(4, 2),
        )

        storage.add_transition(transition)

        assert storage.step == 1
        assert storage.distribution_params is not None
        assert len(storage.distribution_params) == 2

    def test_add_multiple_transitions(self):
        """Test adding multiple transitions."""
        num_envs = 4
        num_transitions = 5
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        torch.manual_seed(42)
        for _ in range(num_transitions):
            transition = Transition()
            transition.observations = torch.randn(num_envs, 8)
            transition.actions = torch.randn(num_envs, 2)
            transition.rewards = torch.randn(num_envs, 1)
            transition.dones = torch.zeros(num_envs, 1)
            transition.values = torch.randn(num_envs, 1)
            transition.actions_log_prob = torch.randn(num_envs, 1)
            transition.distribution_params = (
                torch.randn(num_envs, 2),
                torch.randn(num_envs, 2),
            )

            storage.add_transition(transition)

        assert storage.step == num_transitions

    def test_buffer_overflow(self):
        """Test that buffer overflow raises error."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=2,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        # Fill the buffer
        for _ in range(2):
            transition = Transition()
            transition.observations = torch.randn(4, 8)
            transition.actions = torch.randn(4, 2)
            transition.rewards = torch.randn(4, 1)
            transition.dones = torch.zeros(4, 1)
            transition.values = torch.randn(4, 1)
            transition.actions_log_prob = torch.randn(4, 1)
            transition.distribution_params = (
                torch.randn(4, 2),
                torch.randn(4, 2),
            )
            storage.add_transition(transition)

        # Try to add one more - should raise error
        with pytest.raises(OverflowError):
            transition = Transition()
            transition.observations = torch.randn(4, 8)
            transition.actions = torch.randn(4, 2)
            transition.rewards = torch.randn(4, 1)
            transition.dones = torch.zeros(4, 1)
            transition.values = torch.randn(4, 1)
            transition.actions_log_prob = torch.randn(4, 1)
            transition.distribution_params = (
                torch.randn(4, 2),
                torch.randn(4, 2),
            )
            storage.add_transition(transition)

    def test_clear(self):
        """Test clearing the storage."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        storage.step = 5
        storage.clear()

        assert storage.step == 0

    def test_compute_returns_and_advantages(self):
        """Test computing returns and advantages."""
        num_envs = 4
        num_transitions = 5
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        # Add transitions
        torch.manual_seed(42)
        for _ in range(num_transitions):
            transition = Transition()
            transition.observations = torch.randn(num_envs, 8)
            transition.actions = torch.randn(num_envs, 2)
            transition.rewards = torch.ones(num_envs, 1)  # Constant rewards
            transition.dones = torch.zeros(num_envs, 1)
            transition.values = torch.zeros(num_envs, 1)  # Zero values
            transition.actions_log_prob = torch.randn(num_envs, 1)
            transition.distribution_params = (
                torch.randn(num_envs, 2),
                torch.randn(num_envs, 2),
            )

            storage.add_transition(transition)

        last_values = torch.zeros(num_envs, 1)
        storage.compute_returns_and_advantages(last_values, gamma=0.99, gae_lambda=0.95)

        assert storage.returns is not None
        assert storage.advantages is not None
        assert storage.returns.shape == (num_transitions, num_envs, 1)
        assert storage.advantages.shape == (num_transitions, num_envs, 1)
        assert torch.isfinite(storage.returns).all()
        assert torch.isfinite(storage.advantages).all()

    def test_mini_batch_generator(self):
        """Test mini-batch generator."""
        num_envs = 4
        num_transitions = 6
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        # Add transitions
        torch.manual_seed(42)
        for _ in range(num_transitions):
            transition = Transition()
            transition.observations = torch.randn(num_envs, 8)
            transition.actions = torch.randn(num_envs, 2)
            transition.rewards = torch.randn(num_envs, 1)
            transition.dones = torch.zeros(num_envs, 1)
            transition.values = torch.randn(num_envs, 1)
            transition.actions_log_prob = torch.randn(num_envs, 1)
            transition.distribution_params = (
                torch.randn(num_envs, 2),
                torch.randn(num_envs, 2),
            )

            storage.add_transition(transition)

        # Compute returns and advantages
        storage.compute_returns_and_advantages(torch.zeros(num_envs, 1))

        # Generate mini-batches
        num_mini_batches = 2
        num_epochs = 2
        batch_count = 0

        for batch in storage.mini_batch_generator(num_mini_batches, num_epochs):
            batch_count += 1
            assert batch.observations is not None
            assert batch.actions is not None
            assert batch.values is not None
            assert batch.advantages is not None
            assert batch.returns is not None
            assert batch.old_actions_log_prob is not None
            assert batch.old_distribution_params is not None

        assert batch_count == num_mini_batches * num_epochs

    def test_compute_returns_non_rl_raises(self):
        """Test that computing returns in distillation mode raises error."""
        storage = RolloutBuffer(
            training_type="distillation",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        with pytest.raises(ValueError):
            storage.compute_returns_and_advantages(torch.zeros(4))

    def test_mini_batch_generator_non_rl_raises(self):
        """Test that mini_batch_generator in distillation mode raises error."""
        storage = RolloutBuffer(
            training_type="distillation",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        with pytest.raises(ValueError):
            next(storage.mini_batch_generator(2))


class TestRolloutBufferDistillation:
    """Test RolloutStorage for distillation training."""

    def test_initialization_distillation(self):
        """Test initialization for distillation training."""
        storage = RolloutBuffer(
            training_type="distillation",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        assert storage.training_type == "distillation"
        assert storage.privileged_actions.shape == (10, 4, 2)
        assert not hasattr(storage, "values")

    def test_add_transition_distillation(self):
        """Test adding a transition for distillation."""
        storage = RolloutBuffer(
            training_type="distillation",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        transition = Transition()
        transition.observations = torch.randn(4, 8)
        transition.privileged_actions = torch.randn(4, 2)
        transition.dones = torch.zeros(4)

        storage.add_transition(transition)

        assert storage.step == 1

    def test_distillation_generator(self):
        """Test distillation generator."""
        num_envs = 4
        num_transitions = 5
        storage = RolloutBuffer(
            training_type="distillation",
            num_envs=num_envs,
            num_transitions_per_env=num_transitions,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        # Add transitions
        torch.manual_seed(42)
        for _ in range(num_transitions):
            transition = Transition()
            transition.observations = torch.randn(num_envs, 8)
            transition.privileged_actions = torch.randn(num_envs, 2)
            transition.dones = torch.zeros(num_envs, 1)

            storage.add_transition(transition)

        # Generate batches
        batch_count = 0
        for batch in storage.generator():
            batch_count += 1
            assert batch.observations is not None
            assert batch.privileged_actions is not None
            assert batch.dones is not None

        assert batch_count == num_transitions

    def test_generator_non_distillation_raises(self):
        """Test that generator in RL mode raises error."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=10,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        with pytest.raises(ValueError):
            next(storage.generator())


class TestRolloutBufferMultiDimensional:
    """Test RolloutStorage with multi-dimensional shapes."""

    def test_image_observations(self):
        """Test with image-like observations."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=2,
            num_transitions_per_env=4,
            obs_shape=(3, 32, 32),  # RGB images
            actions_shape=(2,),
            device="cpu",
        )

        assert storage.observations.shape == (4, 2, 3, 32, 32)

    def test_vector_observations(self):
        """Test with vector observations."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=8,
            num_transitions_per_env=10,
            obs_shape=(128,),  # Vector obs
            actions_shape=(64,),  # Vector actions
            device="cpu",
        )

        assert storage.observations.shape == (10, 8, 128)
        assert storage.actions.shape == (10, 8, 64)

    def test_multi_dim_actions(self):
        """Test with multi-dimensional actions."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=5,
            obs_shape=(8,),
            actions_shape=(2, 3),  # 2D action space
            device="cpu",
        )

        assert storage.actions.shape == (5, 4, 2, 3)


class TestRolloutBufferDevice:
    """Test RolloutStorage device placement."""

    def test_cpu_device(self):
        """Test CPU device placement."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=5,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cpu",
        )

        assert storage.observations.device.type == "cpu"
        assert storage.actions.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_device(self):
        """Test CUDA device placement."""
        storage = RolloutBuffer(
            training_type="rl",
            num_envs=4,
            num_transitions_per_env=5,
            obs_shape=(8,),
            actions_shape=(2,),
            device="cuda",
        )

        assert storage.observations.device.type == "cuda"
        assert storage.actions.device.type == "cuda"
