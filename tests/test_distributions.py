"""Tests for distribution modules."""

import math
import torch

from telekinesis.rlbotics.distributions import (
    GaussianDistribution,
    SquashedGaussianDistribution,
)


class TestGaussianDistribution:
    """Test GaussianDistribution with state-independent std."""

    def test_init_default(self):
        """Test default initialization."""
        dist = GaussianDistribution(act_dim=4)
        assert dist.act_dim == 4
        assert hasattr(dist, "log_std_param")

    def test_forward_pass(self):
        """Test basic forward pass."""
        dist = GaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.randn(2, 4)
        dist.update(mean)

        sample = dist.sample()
        assert sample.shape == (2, 4)
        assert torch.isfinite(sample).all()

    def test_log_prob_standard_normal(self):
        """Log prob at mean of N(0,1) equals -0.5*log(2π) per dimension."""
        dim = 4
        dist = GaussianDistribution(act_dim=dim, init_std=1.0)
        mean = torch.zeros(1, dim)
        dist.update(mean)

        log_p = dist.log_prob(torch.zeros(1, dim))
        expected = -0.5 * math.log(2 * math.pi) * dim
        assert torch.allclose(log_p, torch.tensor([expected]), atol=1e-5)

    def test_log_prob_nonzero_mean(self):
        """Log prob should decrease as sample moves away from mean."""
        dist = GaussianDistribution(act_dim=2, init_std=1.0)
        mean = torch.tensor([[3.0, 3.0]])
        dist.update(mean)

        lp_at_mean = dist.log_prob(mean)
        lp_far = dist.log_prob(mean + 5.0)
        assert lp_at_mean > lp_far, "log_prob higher at mean"
        assert torch.isfinite(lp_at_mean).all()
        assert torch.isfinite(lp_far).all()

    def test_entropy_analytical(self):
        """Entropy matches formula 0.5*sum(log(2πe*std²))."""
        dim = 3
        std_val = 2.0
        dist = GaussianDistribution(act_dim=dim, init_std=std_val)
        dist.update(torch.zeros(1, dim))

        expected = 0.5 * dim * math.log(2 * math.pi * math.e * std_val**2)
        assert torch.allclose(dist.entropy, torch.tensor([expected]), atol=1e-4)

    def test_entropy_positive(self):
        """Entropy should always be positive."""
        dist = GaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.randn(2, 4)
        dist.update(mean)

        entropy = dist.entropy
        assert entropy.shape == (2,)
        assert (entropy > 0).all()
        assert torch.isfinite(entropy).all()

    def test_kl_divergence_analytical(self):
        """KL(N(0,1) || N(μ,σ)) matches closed form."""
        dist = GaussianDistribution(act_dim=1, init_std=1.0)
        mu_old, sigma_old = 0.0, 1.0
        mu_new, sigma_new = 1.0, 2.0

        old_params = (torch.tensor([[mu_old]]), torch.tensor([[sigma_old]]))
        new_params = (torch.tensor([[mu_new]]), torch.tensor([[sigma_new]]))

        kl = dist.kl_divergence(old_params, new_params)

        expected = (math.log(sigma_new / sigma_old) +
                   (sigma_old**2 + (mu_old - mu_new) ** 2) / (2 * sigma_new**2) - 0.5)
        assert torch.allclose(kl, torch.tensor([expected]), atol=1e-5)

    def test_kl_divergence_identical_is_zero(self):
        """KL(p || p) should be zero."""
        dist = GaussianDistribution(act_dim=4, init_std=1.5)
        params = (torch.zeros(1, 4), torch.full((1, 4), 1.5))
        kl = dist.kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros(1), atol=1e-6)

    def test_kl_divergence_nonnegative(self):
        """KL divergence should always be non-negative."""
        dist = GaussianDistribution(act_dim=4)
        mean1 = torch.randn(1, 4)
        std1 = torch.abs(torch.randn(1, 4)) + 0.1
        mean2 = torch.randn(1, 4)
        std2 = torch.abs(torch.randn(1, 4)) + 0.1

        params1 = (mean1, std1)
        params2 = (mean2, std2)

        kl = dist.kl_divergence(params1, params2)
        assert (kl >= 0).all()
        assert torch.isfinite(kl).all()

    def test_params_returns_mean_and_std(self):
        """Params property should return mean and std."""
        dist = GaussianDistribution(act_dim=4)
        mean = torch.randn(2, 4)
        dist.update(mean)

        params = dist.params
        assert len(params) == 2
        assert torch.allclose(params[0], dist.mean)
        assert torch.allclose(params[1], dist.std)

    def test_deterministic_output(self):
        """Deterministic output extraction."""
        dist = GaussianDistribution(act_dim=4)
        mean = torch.randn(2, 4)
        dist.update(mean)

        det_output = dist.deterministic_output(mean)
        assert torch.allclose(det_output, mean)

    def test_std_clamping_upper(self):
        """Std is clamped to upper bound."""
        dist = GaussianDistribution(act_dim=4, init_std=100.0, std_range=(0.1, 1.0))
        mean = torch.zeros(1, 4)
        dist.update(mean)

        std = dist.std
        assert torch.allclose(std, torch.full((1, 4), 1.0), atol=1e-6)

    def test_std_clamping_lower(self):
        """Std is clamped to lower bound."""
        dist = GaussianDistribution(act_dim=4, init_std=1e-8, std_range=(0.1, 1.0))
        mean = torch.zeros(1, 4)
        dist.update(mean)

        std = dist.std
        assert torch.allclose(std, torch.full((1, 4), 0.1), atol=1e-6)

    def test_std_range_min_floor(self):
        """Minimum of std_range is floored to 1e-6."""
        dist = GaussianDistribution(act_dim=2, init_std=1.0, std_range=(0.0, 10.0))
        assert dist.std_range[0] == 1e-6

    def test_input_dim_property(self):
        """Input dim property matches act_dim."""
        dist = GaussianDistribution(act_dim=8)
        assert dist.input_dim == 8

    def test_as_deterministic_output_module(self):
        """Export-friendly deterministic output module."""
        dist = GaussianDistribution(act_dim=4)
        module = dist.as_deterministic_output_module()

        mean = torch.randn(2, 4)
        output = module(mean)
        assert torch.allclose(output, mean)

    def test_init_mlp_weights(self):
        """MLP weight initialization hook."""
        dist = GaussianDistribution(act_dim=4)
        mlp = torch.nn.Linear(8, 4)
        dist.init_mlp_weights(mlp)

    def test_log_prob_gradient_flows(self):
        """Gradient flows from log_prob back to distribution."""
        dim = 3
        dist = GaussianDistribution(act_dim=dim, init_std=1.0)
        mean = torch.randn(1, dim, requires_grad=True)
        dist.update(mean)

        sample = dist.sample().detach()
        log_p = dist.log_prob(sample)
        log_p.sum().backward()

        assert mean.grad is not None
        assert not torch.all(mean.grad == 0)

    def test_batch_independence(self):
        """Batch samples should be independent."""
        dist = GaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.zeros(32, 4)
        dist.update(mean)

        samples = dist.sample()
        assert samples.shape == (32, 4)
        assert not torch.allclose(samples[0], samples[1])

    def test_numerical_stability_large_std(self):
        """Numerically stable with large std."""
        dist = GaussianDistribution(act_dim=4, init_std=100.0)
        mean = torch.zeros(1, 4)
        dist.update(mean)

        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        assert torch.isfinite(log_prob).all()

    def test_numerical_stability_small_std(self):
        """Numerically stable with small std."""
        dist = GaussianDistribution(act_dim=4, init_std=1e-6)
        mean = torch.zeros(1, 4)
        dist.update(mean)

        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        assert torch.isfinite(log_prob).all()


class TestSquashedGaussianDistribution:
    """Test SquashedGaussianDistribution with tanh squashing."""

    def test_sample_bounds(self):
        """Samples are bounded in [-1, 1]."""
        dist = SquashedGaussianDistribution(act_dim=8, init_std=1.0)
        mean = torch.randn(4, 8)
        dist.update(mean)

        for _ in range(10):
            sample = dist.sample()
            assert (sample >= -1.0).all()
            assert (sample <= 1.0).all()

    def test_sample_shape(self):
        """Sample shape matches input."""
        dist = SquashedGaussianDistribution(act_dim=6, init_std=0.5)
        mean = torch.randn(3, 6)
        dist.update(mean)

        sample = dist.sample()
        assert sample.shape == (3, 6)

    def test_log_prob_with_jacobian(self):
        """Log prob includes Jacobian correction."""
        dist = SquashedGaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.zeros(1, 4)
        dist.update(mean)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        assert not torch.isnan(log_prob).any()
        assert not torch.isinf(log_prob).any()
        assert torch.isfinite(log_prob).all()

    def test_log_prob_boundary_values(self):
        """Log prob is finite at near-boundary values."""
        dist = SquashedGaussianDistribution(act_dim=4)
        mean = torch.zeros(1, 4)
        dist.update(mean)

        action = torch.tensor([[0.99999, -0.99999, 0.5, -0.5]])
        log_prob = dist.log_prob(action)

        assert not torch.isnan(log_prob).any()
        assert not torch.isinf(log_prob).any()

    def test_entropy_approximation(self):
        """Entropy is approximated from base Gaussian."""
        dist = SquashedGaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.zeros(2, 4)
        dist.update(mean)

        entropy = dist.entropy
        assert entropy.shape == (2,)
        assert (entropy > 0).all()
        assert torch.isfinite(entropy).all()

    def test_inheritance(self):
        """SquashedGaussian inherits from Gaussian."""
        dist = SquashedGaussianDistribution(act_dim=4)
        assert isinstance(dist, GaussianDistribution)

    def test_log_prob_is_finite(self):
        """Log prob is finite for valid squashed actions."""
        dist = SquashedGaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.zeros(1, 4)
        dist.update(mean)

        center = torch.zeros(1, 4)
        edge = torch.ones(1, 4) * 0.5

        log_prob_center = dist.log_prob(center)
        log_prob_edge = dist.log_prob(edge)

        assert torch.isfinite(log_prob_center).all()
        assert torch.isfinite(log_prob_edge).all()

    def test_deterministic_output_is_squashed(self):
        """The deterministic action is the squashed mean, matching what sampling produces.

        Returning the pre-tanh mean would have a deployed policy emit actions outside the range it
        trained in, and the further the mean drifts from zero the larger the mismatch.
        """
        dist = SquashedGaussianDistribution(act_dim=4)
        mean = torch.randn(2, 4)
        dist.update(mean)

        det_output = dist.deterministic_output(mean)

        assert torch.allclose(det_output, torch.tanh(mean))
        assert det_output.abs().max() < 1.0

    def test_deterministic_output_stays_bounded_for_a_drifted_mean(self):
        """A large pre-tanh mean still yields an action inside the range, saturating rather than
        escaping it."""
        dist = SquashedGaussianDistribution(act_dim=4)
        mean = torch.full((2, 4), 8.0)
        dist.update(mean)

        assert dist.deterministic_output(mean).abs().max() < 1.0

    def test_export_module_squashes_too(self):
        """The exported graph carries the same squashing, so ONNX and torch agree."""
        dist = SquashedGaussianDistribution(act_dim=4)
        mean = torch.full((2, 4), 3.0)
        dist.update(mean)

        module = dist.as_deterministic_output_module()

        assert torch.allclose(module(mean), dist.deterministic_output(mean))

    def test_gradient_flow(self):
        """Gradient flows through sampling."""
        dist = SquashedGaussianDistribution(act_dim=4)
        mean = torch.randn(2, 4, requires_grad=True)
        dist.update(mean)

        sample = dist.sample()
        loss = sample.sum()
        loss.backward()

        assert mean.grad is not None
        assert not torch.isnan(mean.grad).any()


class TestDistributionBatching:
    """Test proper batching across distributions."""

    def test_gaussian_batch_shape(self):
        """Log prob batch shape matches input."""
        dist = GaussianDistribution(act_dim=4)
        mean = torch.randn(16, 4)
        dist.update(mean)

        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        assert log_prob.shape == (16,)

    def test_squashed_batch_shape(self):
        """Log prob batch shape for SquashedGaussian."""
        dist = SquashedGaussianDistribution(act_dim=4)
        mean = torch.randn(16, 4)
        dist.update(mean)

        sample = dist.sample()
        log_prob = dist.log_prob(sample)

        assert log_prob.shape == (16,)

    def test_entropy_batch_shape(self):
        """Entropy batch shape matches input."""
        dist = GaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.randn(8, 4)
        dist.update(mean)

        entropy = dist.entropy
        assert entropy.shape == (8,)


class TestDistributionConsistency:
    """Test consistency across different operations."""

    def test_log_prob_sum_over_dims(self):
        """Log prob sums over last dimension correctly."""
        dist = GaussianDistribution(act_dim=4, init_std=1.0)
        mean = torch.zeros(1, 4)
        dist.update(mean)

        sample = torch.zeros(1, 4)
        log_prob = dist.log_prob(sample)

        # Should be 4 times the per-dimension log prob of N(0,1)
        expected_per_dim = -0.5 * math.log(2 * math.pi)
        expected_total = expected_per_dim * 4
        assert torch.allclose(log_prob, torch.tensor([expected_total]), atol=1e-5)

    def test_deterministic_and_stochastic_have_same_mean(self):
        """Mean from deterministic and stochastic match."""
        dist = GaussianDistribution(act_dim=4)
        mean = torch.randn(2, 4)
        dist.update(mean)

        det_out = dist.deterministic_output(mean)
        assert torch.allclose(det_out, dist.mean)

    def test_params_stability(self):
        """Params remain stable across multiple updates."""
        dist = GaussianDistribution(act_dim=4)
        mean1 = torch.randn(2, 4)
        dist.update(mean1)
        params1 = dist.params

        mean2 = torch.randn(2, 4)
        dist.update(mean2)
        params2 = dist.params

        # Should not be the same (different inputs)
        assert not torch.allclose(params1[0], params2[0])
