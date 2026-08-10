"""Normalization modules for observations and rewards."""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        until: int | None = None,
    ) -> None:
        """Initialize EmpiricalNormalization module.

        The normalization parameters are computed over the whole batch, not for each environment
        separately.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            until: If this arg is specified, the module learns input values until the sum of batch
                sizes exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self) -> torch.Tensor:
        """Return the current running mean."""
        return self._mean.squeeze(0).clone()

    @property
    def std(self) -> torch.Tensor:
        """Return the current running standard deviation."""
        return self._std.squeeze(0).clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize mean and variance of values based on empirical values."""
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x: torch.Tensor) -> None:
        """Learn input values without computing the output values of them.

        Args:
            x: Input tensor of shape (batch_size, *shape).
        """
        if not self.training:
            return
        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count
        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """De-normalize values based on empirical values.

        Args:
            y: Normalized tensor.

        Returns:
            De-normalized tensor.
        """
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to
    normalize the scale of the rewards so that the value function can learn quickly.
    We did this by dividing the rewards by a running estimate of the standard deviation
    of the sum of discounted rewards.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-2,
        gamma: float = 0.99,
        until: int | None = None,
    ) -> None:
        """Initialize discounted-reward normalization with running moments.

        Args:
            shape: Shape of input values except batch axis.
            eps: Small value for stability.
            gamma: Discount factor for reward accumulation.
            until: If specified, stops learning after sum of batch sizes exceeds it.
        """
        super().__init__()
        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = _DiscountedAverage(gamma)

    def forward(self, rew: torch.Tensor) -> torch.Tensor:
        """Normalize rewards using the running std of discounted returns.

        Args:
            rew: Reward tensor of shape (batch_size, *shape).

        Returns:
            Normalized reward tensor.
        """
        if self.training:
            # Update discounted rewards
            avg = self.disc_avg.update(rew)
            # Update moments from discounted rewards
            self.emp_norm.update(avg)

        # Normalize rewards with the empirical std
        if torch.any(self.emp_norm._std > 0):
            return rew / self.emp_norm._std
        return rew


class _DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t
    """

    def __init__(self, gamma: float) -> None:
        """Initialize discounted accumulation with a fixed discount factor.

        Args:
            gamma: Discount factor.
        """
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Update and return the discounted running average.

        Args:
            rew: Reward tensor.

        Returns:
            Discounted average reward tensor.
        """
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg
