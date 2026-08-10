"""Symmetry data augmentation and mirror loss for PPO.

A robot that is left-right symmetric gives the algorithm an invariance for free, and this extension is
how the algorithm is told about it. Kept in its own module because the mirror function it needs comes
from outside the library — from whoever knows the robot's joint order — so this is the seam where user
code meets the update loop.

The settings live in :class:`~telekinesis.rlbotics.config.SymmetryConfig`, and
:class:`~telekinesis.rlbotics.algorithms.PPO` builds this when one is given.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from loguru import logger

from telekinesis.rlbotics.config import SymmetryConfig
from telekinesis.rlbotics.models import CNNModel, MLPModel
from telekinesis.rlbotics.utils import resolve_callable


class Symmetry:
    """Symmetry data augmentation and mirror loss.

    A legged robot is left-right symmetric, so a policy that has learned to trot leading with the
    left leg has, in principle, learned the mirrored gait too. Telling the algorithm about that
    symmetry is worth real sample efficiency, and it is what stops a policy settling into a limp.

    Both uses of the mirror function are optional and independent:

    - ``use_data_augmentation`` appends the mirrored observation and action pairs to every
      mini-batch, so the surrogate and value losses see both.
    - ``use_mirror_loss`` adds an auxiliary term penalizing the policy for disagreeing with itself
      on mirrored observations.

    With neither on, the loss is still computed and reported, detached from the graph, which is a
    cheap way to watch how symmetric a policy is without changing what it optimizes.

    The mirror function is supplied by whoever knows the robot's joint layout, since only they can
    say which observation entry mirrors which. It is called as
    ``func(env=env, obs=obs, actions=actions)`` and returns the pair, each stacked as
    ``[original, mirrored, ...]`` along the batch dimension. Either argument may be None, in which
    case the corresponding return is ignored.

    References:
        Mittal et al., "Symmetry Considerations for Learning Task Symmetric Robot Policies",
        ICRA 2024.
    """

    def __init__(self, cfg: SymmetryConfig, env=None) -> None:
        """Initialize the extension from its config.

        Args:
            cfg: Symmetry configuration, holding the mirror function and what to use it for.
            env: Environment handed to the mirror function, which usually needs it to know the
                observation layout. Defaults to None; the runner passes the environment it trains on.
        """
        self.cfg = cfg
        self.env = env
        self.data_augmentation_func = resolve_callable(cfg.data_augmentation_func)

        if not (cfg.use_data_augmentation or cfg.use_mirror_loss):
            logger.info(
                "   Symmetry is configured with neither data augmentation nor the mirror loss, so "
                "it only reports how symmetric the policy is."
            )

    def augment_batch(self, batch, original_batch_size: int) -> None:
        """Append the mirrored samples to a mini-batch, in place.

        Afterwards the observations and actions hold ``original_batch_size * num_aug`` rows, the
        originals first, and every other rollout tensor is repeated to match so the losses line up.
        Does nothing when data augmentation is off.

        Args:
            batch: The mini-batch to augment.
            original_batch_size: Rows the batch had before augmenting.
        """
        if not self.cfg.use_data_augmentation:
            return

        batch.observations, batch.actions = self.data_augmentation_func(
            env=self.env, obs=batch.observations, actions=batch.actions
        )
        num_aug = int(batch.observations.shape[0] / original_batch_size)

        # A critic that reads its own observation set has to grow with the actor's, or the value loss
        # would compare a batch of values against a batch of returns twice its size
        critic_obs = getattr(batch, "critic_observations", None)
        if critic_obs is not None and critic_obs.shape[0] == original_batch_size:
            batch.critic_observations, _ = self.data_augmentation_func(
                env=self.env, obs=critic_obs, actions=None
            )

        for name in ("old_actions_log_prob", "values", "advantages", "returns"):
            tensor = getattr(batch, name, None)
            if tensor is not None:
                setattr(batch, name, tensor.repeat(num_aug, *([1] * (tensor.dim() - 1))))

    def compute_loss(self, actor: MLPModel | CNNModel, batch, original_batch_size: int) -> torch.Tensor:
        """Return the mirror loss: how far the policy is from being symmetric on this batch.

        The comparison is between the action means the actor predicts on mirrored observations and
        the mirror of the means it predicts on the original ones. Means rather than sampled actions,
        because the symmetry is a property of the policy, not of the noise.

        Args:
            actor: The policy being trained.
            batch: The mini-batch, already augmented if data augmentation is on.
            original_batch_size: Rows the batch had before augmenting.

        Returns:
            The mirror loss, detached when it is only being reported.
        """
        # Without data augmentation the batch is still one-sided, so mirror it here
        if not self.cfg.use_data_augmentation:
            batch.observations, _ = self.data_augmentation_func(
                env=self.env, obs=batch.observations, actions=None
            )

        mean_actions = actor(batch.observations.detach().clone(), stochastic=False)
        _, mean_actions_mirrored = self.data_augmentation_func(
            env=self.env, obs=None, actions=mean_actions[:original_batch_size]
        )

        symmetry_loss = nn.functional.mse_loss(
            mean_actions[original_batch_size:],
            mean_actions_mirrored.detach()[original_batch_size:],
        )
        return symmetry_loss if self.cfg.use_mirror_loss else symmetry_loss.detach()
