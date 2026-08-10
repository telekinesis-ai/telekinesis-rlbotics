"""Tests for normalization modules."""

import pytest
import torch
import torch.nn as nn
from telekinesis.rlbotics.normalization import (
    EmpiricalNormalization,
    EmpiricalDiscountedVariationNormalization,
)


class TestEmpiricalNormalization:
    """Test EmpiricalNormalization module."""

    def test_initialization_scalar_shape(self):
        """Test initialization with scalar shape."""
        norm = EmpiricalNormalization(shape=10, eps=1e-2)
        assert norm.mean.shape == (10,)
        assert norm.std.shape == (10,)
        assert norm.count == 0
        assert norm.eps == 1e-2

    def test_initialization_tuple_shape(self):
        """Test initialization with tuple shape."""
        norm = EmpiricalNormalization(shape=(3, 32, 32), eps=1e-2)
        assert norm.mean.shape == (3, 32, 32)
        assert norm.std.shape == (3, 32, 32)

    def test_initialization_list_shape(self):
        """Test initialization with list shape."""
        norm = EmpiricalNormalization(shape=[4, 5], eps=1e-2)
        assert norm.mean.shape == (4, 5)
        assert norm.std.shape == (4, 5)

    def test_forward_normalization(self):
        """Test forward normalization."""
        norm = EmpiricalNormalization(shape=10, eps=1e-2)
        norm.train()

        # Create data with known statistics
        x = torch.randn(100, 10)
        norm.update(x)

        # Normalize the same data
        normalized = norm(x)

        # Normalized data should have approximately zero mean and unit variance
        assert torch.abs(normalized.mean(dim=0)).max() < 0.3
        assert torch.abs(normalized.std(dim=0).mean() - 1.0) < 0.2

    def test_update_single_batch(self):
        """Test update with single batch."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2)
        norm.train()

        x = torch.randn(20, 5)
        norm.update(x)

        assert norm.count == 20
        assert torch.allclose(norm.mean, x.mean(dim=0), atol=1e-5)
        assert torch.allclose(norm.std, torch.sqrt(x.var(dim=0, unbiased=False)), atol=1e-5)

    def test_update_multiple_batches(self):
        """Test update with multiple batches."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2)
        norm.train()

        batches = [torch.randn(10, 5) for _ in range(5)]
        for batch in batches:
            norm.update(batch)

        # After all updates, count should be 50
        assert norm.count == 50

        # Combined statistics should match all data
        all_data = torch.cat(batches, dim=0)
        assert torch.allclose(norm.mean, all_data.mean(dim=0), atol=1e-5)
        assert torch.allclose(norm.std, torch.sqrt(all_data.var(dim=0, unbiased=False)), atol=1e-5)

    def test_no_update_in_eval_mode(self):
        """Test that update is ignored in evaluation mode."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2)

        # Training mode - update
        norm.train()
        x1 = torch.randn(10, 5)
        norm.update(x1)
        count_after_train = norm.count.item()

        # Evaluation mode - update should be ignored
        norm.eval()
        x2 = torch.randn(10, 5)
        norm.update(x2)

        assert norm.count == count_after_train

    def test_inverse_operation(self):
        """Test denormalization (inverse) operation."""
        norm = EmpiricalNormalization(shape=10, eps=1e-2)
        norm.train()

        x = torch.randn(50, 10)
        norm.update(x)
        norm.eval()

        # Normalize and denormalize
        normalized = norm(x)
        denormalized = norm.inverse(normalized)

        # Should recover original values
        assert torch.allclose(denormalized, x, atol=1e-5)

    def test_until_parameter(self):
        """Test learning cutoff with 'until' parameter."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2, until=50)
        norm.train()

        # First update (count < 50, so learning happens)
        x1 = torch.randn(30, 5)
        norm.update(x1)
        assert norm.count == 30

        # Second update (count >= 50 check happens after increment)
        x2 = torch.randn(30, 5)
        mean_after_first = norm.mean.clone()
        norm.update(x2)  # count becomes 60, but still accepts this batch
        assert norm.count == 60

        # Third update (count >= 50, so no learning happens)
        x3 = torch.randn(30, 5)
        mean_after_second = norm.mean.clone()
        norm.update(x3)  # This update is actually ignored
        assert norm.count == 60  # Count doesn't change
        assert torch.allclose(norm.mean, mean_after_second)  # Mean doesn't change

    def test_properties(self):
        """Test mean and std properties."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2)
        norm.train()

        x = torch.randn(30, 5)
        norm.update(x)

        # Properties should return cloned tensors
        mean1 = norm.mean
        mean2 = norm.mean
        assert torch.allclose(mean1, mean2)
        assert mean1 is not mean2  # Different objects

        std1 = norm.std
        std2 = norm.std
        assert torch.allclose(std1, std2)
        assert std1 is not std2  # Different objects

    def test_eps_stability(self):
        """Test that eps prevents division by zero."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2)
        norm.train()

        # Update with zero variance data
        x = torch.ones(10, 5)
        norm.update(x)

        # Forward pass should not produce NaN or Inf
        output = norm(x)
        assert torch.isfinite(output).all()

    def test_batched_shape_support(self):
        """Test support for batched shape in multi-dimensional inputs."""
        norm = EmpiricalNormalization(shape=(3, 4), eps=1e-2)
        norm.train()

        # Batch of 2D tensors
        x = torch.randn(16, 3, 4)
        norm.update(x)

        assert norm.count == 16
        assert norm.mean.shape == (3, 4)
        assert norm.std.shape == (3, 4)

        # Normalize should work correctly
        normalized = norm(x)
        assert normalized.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through normalization."""
        norm = EmpiricalNormalization(shape=5, eps=1e-2)
        norm.train()

        x_train = torch.randn(50, 5)
        norm.update(x_train)
        norm.eval()

        # Test gradient flow
        x = torch.randn(4, 5, requires_grad=True)
        y = norm(x)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestEmpiricalDiscountedVariationNormalization:
    """Test EmpiricalDiscountedVariationNormalization module."""

    def test_initialization(self):
        """Test initialization."""
        norm = EmpiricalDiscountedVariationNormalization(
            shape=1, eps=1e-2, gamma=0.99
        )
        assert isinstance(norm.emp_norm, EmpiricalNormalization)
        assert norm.emp_norm.eps == 1e-2

    def test_forward_training_mode(self):
        """Test forward pass in training mode updates statistics."""
        norm = EmpiricalDiscountedVariationNormalization(
            shape=1, eps=1e-2, gamma=0.99
        )
        norm.train()

        # Apply rewards sequentially
        for step in range(5):
            reward = torch.tensor([[0.1]], dtype=torch.float32)
            normalized = norm(reward)
            assert normalized.shape == reward.shape

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode doesn't update."""
        norm = EmpiricalDiscountedVariationNormalization(
            shape=1, eps=1e-2, gamma=0.99
        )
        norm.train()

        # Train on some rewards
        for _ in range(5):
            reward = torch.tensor([[0.1]], dtype=torch.float32)
            norm(reward)

        std_after_train = norm.emp_norm.std.clone()

        # Switch to eval and try to update (should not change)
        norm.eval()
        for _ in range(5):
            reward = torch.tensor([[10.0]], dtype=torch.float32)
            norm(reward)

        assert torch.allclose(norm.emp_norm.std, std_after_train)

    def test_normalization_scaling(self):
        """Test that normalization scales rewards appropriately."""
        norm = EmpiricalDiscountedVariationNormalization(
            shape=1, eps=1e-2, gamma=0.99
        )
        norm.train()

        # Apply increasing rewards to build up std
        for i in range(20):
            reward = torch.tensor([[float(i)]], dtype=torch.float32)
            norm(reward)

        # After training, rewards should be normalized by std
        norm.eval()
        small_reward = torch.tensor([[1.0]], dtype=torch.float32)
        normalized = norm(small_reward)

        # Normalized reward should be smaller due to division by std
        assert torch.abs(normalized).item() < torch.abs(small_reward).item()

    def test_different_shapes(self):
        """Test with different input shapes."""
        for shape in [1, (2,), (3, 4), (2, 3, 4)]:
            norm = EmpiricalDiscountedVariationNormalization(
                shape=shape, eps=1e-2, gamma=0.99
            )
            norm.train()

            if isinstance(shape, int):
                batch_shape = (16, shape)
            else:
                batch_shape = (16,) + shape

            reward = torch.randn(batch_shape)
            output = norm(reward)

            assert output.shape == reward.shape


class TestNormalizationIntegration:
    """Integration tests for normalization modules."""

    def test_multiple_normalizers(self):
        """Test using multiple normalizers together."""
        obs_norm = EmpiricalNormalization(shape=10, eps=1e-2)
        reward_norm = EmpiricalDiscountedVariationNormalization(
            shape=1, eps=1e-2, gamma=0.99
        )

        obs_norm.train()
        reward_norm.train()

        # Simulate multiple steps
        for _ in range(5):
            obs = torch.randn(32, 10)
            reward = torch.randn(32, 1)

            obs_norm.update(obs)
            reward_norm(reward)

        # Both should be in valid state
        assert obs_norm.count > 0
        assert torch.isfinite(obs_norm.mean).all()
        assert torch.isfinite(reward_norm.emp_norm.mean).all()

    def test_state_dict(self):
        """Test saving and loading state."""
        norm1 = EmpiricalNormalization(shape=5, eps=1e-2)
        norm1.train()

        x = torch.randn(50, 5)
        norm1.update(x)

        # Save state
        state_dict = norm1.state_dict()

        # Create new normalizer and load state
        norm2 = EmpiricalNormalization(shape=5, eps=1e-2)
        norm2.load_state_dict(state_dict)

        # Both should have same statistics
        assert torch.allclose(norm1.mean, norm2.mean)
        assert torch.allclose(norm1.std, norm2.std)
        assert norm1.count == norm2.count

    def test_convergence_to_known_distribution(self):
        """Running mean and std converge to true distribution values."""
        true_mean, true_std = 5.0, 2.0
        norm = EmpiricalNormalization(shape=4, eps=1e-2)
        norm.train()

        torch.manual_seed(0)
        for _ in range(200):
            batch = true_mean + true_std * torch.randn(64, 4)
            norm.update(batch)

        assert torch.allclose(norm.mean, torch.full((4,), true_mean), atol=0.2)
        assert torch.allclose(norm.std, torch.full((4,), true_std), atol=0.2)
