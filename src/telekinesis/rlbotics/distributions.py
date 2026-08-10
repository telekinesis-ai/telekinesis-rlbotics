"""Distribution modules for policy parameterization in reinforcement learning."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Distribution(nn.Module):
    """Base class for distribution modules.

    Distribution modules encapsulate the stochastic output of a neural model. They define the output
    structure expected from the MLP, manage learnable distribution parameters, and provide methods
    for sampling, log probability computation, and entropy calculation.

    Subclasses must implement all abstract methods and properties to define a specific distribution
    type.
    """

    def __init__(self, act_dim: int) -> None:
        """Initialize the distribution module.

        Args:
            act_dim: Dimension of the action/output space.
        """
        super().__init__()
        self.act_dim = act_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution parameters given the MLP output.

        Args:
            mlp_output: Raw output from the MLP.
        """
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """Sample from the distribution.

        Returns:
            Sampled values.
        """
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the deterministic (mean) output from the raw MLP output.

        Args:
            mlp_output: Raw output from the MLP.

        Returns:
            The deterministic output (typically the distribution mean).
        """
        raise NotImplementedError

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the deterministic output from the MLP

        output.
        """
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        """Return the input dimension required by the distribution."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation (or spread measure) of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the distribution, summed over the last dimension."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the distribution parameters as a tuple of tensors.

        These are the distribution-specific parameters needed to reconstruct the distribution (e.g.,
        mean and std
        for Gaussian, alpha and beta for Beta). They are stored during rollouts and used for KL
        divergence computation.
        """
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given outputs, summed over the last dimension.

        Args:
            outputs: Values to compute the log probability for.

        Returns:
            Log probability summed over the last dimension.
        """
        raise NotImplementedError

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute the KL divergence KL(old || new) between two distributions of this type.

        The KL divergence measures how the old distribution diverges from the new distribution. This
        is used for adaptive learning rate scheduling in policy optimization.

        Args:
            old_params: Parameters of the old distribution (as returned by :attr:`params`).
            new_params: Parameters of the new distribution (as returned by :attr:`params`).

        Returns:
            KL divergence summed over the last dimension.
        """
        raise NotImplementedError

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize distribution-specific weights in the MLP.

        This is called after MLP creation to set up any special weight initialization required by
        the distribution (e.g., initializing std head weights).

        Args:
            mlp: The MLP module whose weights may need initialization.
        """
        pass


class GaussianDistribution(Distribution):
    """Gaussian distribution with state-independent standard deviation.

    Parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation
    can be learnable or fixed, and is clamped to a specified range for numerical stability.
    """

    def __init__(
        self,
        act_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            act_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_range: Range for standard deviation clamping (min, max). Defaults to (1e-6, 1e6).
        """
        super().__init__(act_dim)

        # Create learnable log std parameter (note: log std is used for numerical stability)
        log_init_std = float(np.log(init_std))
        self.log_std_param = nn.Parameter(log_init_std * torch.ones(act_dim))

        # Clamp std range for numerical stability
        self.std_range = [max(std_range[0], 1e-6), std_range[1]]
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        self._distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution from MLP output (mean)."""
        mean = mlp_output
        log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
        std = torch.exp(log_std)

        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean (deterministic output) from MLP output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module for deterministic output."""
        return _IdentityDeterministicOutput()

    @property
    def input_dim(self) -> int:
        """Return input dimension required by the distribution."""
        return self.act_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Gaussian distribution."""
        return self._distribution.mean  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Gaussian distribution."""
        return self._distribution.stddev  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) of the current distribution."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log probability, summed over the last dimension."""
        return self._distribution.log_prob(outputs).sum(dim=-1)  # type: ignore

    def kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL(old || new) between two Gaussian distributions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        return torch.distributions.kl_divergence(
            Normal(old_mean, old_std), Normal(new_mean, new_std)
        ).sum(dim=-1)


class SquashedGaussianDistribution(GaussianDistribution):
    """Gaussian distribution with tanh squashing for bounded continuous control.

    Samples from a Gaussian and applies tanh squashing to produce bounded outputs in [-1, 1].
    Correctly computes log probabilities with Jacobian correction.

    This is the standard distribution for continuous control in modern RL algorithms (SAC, PPO,
    TRPO).
    """

    def sample(self) -> torch.Tensor:
        """Sample from Gaussian and apply tanh squashing."""
        z = self._distribution.rsample()  # type: ignore
        return torch.tanh(z)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log probability with tanh Jacobian correction.

        Args:
            outputs: Tanh-squashed actions in [-1, 1].

        Returns:
            Log probability with Jacobian correction, summed over last dimension.
        """
        # Inverse tanh: z = 0.5 * log((1 + y) / (1 - y))
        outputs_clamped = torch.clamp(outputs, -0.999999, 0.999999)
        z = 0.5 * torch.log((1 + outputs_clamped) / (1 - outputs_clamped))

        # Log prob of z under Gaussian
        log_prob = self._distribution.log_prob(z).sum(dim=-1)  # type: ignore

        # Jacobian correction: log p(y) = log p(z) - sum(log(1 - tanh(z)^2))
        log_prob -= torch.log(1 - outputs.pow(2) + 1e-6).sum(dim=-1)

        return log_prob

    @property
    def entropy(self) -> torch.Tensor:
        """Return approximate entropy (tanh-squashed Gaussian entropy is intractable)."""
        # Use base Gaussian entropy as approximation
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the squashed mean, which is the action this policy takes deterministically.

        The pre-tanh mean is not an action: sampling passes it through tanh, so returning it raw
        would have a deployed policy emit values outside the range it was trained in, and the further
        the mean drifts the worse the mismatch. This is also what the exported graph carries, via
        :meth:`as_deterministic_output_module`.

        Args:
            mlp_output: Raw output from the MLP, the pre-tanh mean.

        Returns:
            The action, bounded to (-1, 1).
        """
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module for the squashed deterministic output."""
        return _TanhDeterministicOutput()


class _IdentityDeterministicOutput(nn.Module):
    """Returns MLP output as-is (for Gaussian mean)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _TanhDeterministicOutput(nn.Module):
    """Squashes the MLP output, matching how a squashed Gaussian samples."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output)
