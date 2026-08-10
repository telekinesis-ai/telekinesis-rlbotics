"""Loading a trained policy for inference, without torch."""

from __future__ import annotations

from pathlib import Path

import numpy as np


class Policy:
    """A trained policy loaded from ONNX, for acting in an environment.

    This is the deployment side of training: it needs neither torch nor any part of the training
    stack, only :mod:`onnxruntime` and numpy. Everything the policy needs is inside the exported
    graph, including the observation normalization it was trained with and the scaling onto the
    environment's action range, so raw observations go in and environment-ready actions come out.

    Example:
        Load an exported policy and act with it::

            policy = Policy("logs/experiment/2026-01-31_10-15-00/policy.onnx")
            action = policy.get_action(observation)
    """

    def __init__(self, path: str | Path) -> None:
        """Load an exported policy.

        Args:
            path: Path to a ``.onnx`` file written by
                :meth:`~telekinesis.rlbotics.runner.OnPolicyRunner.export`.

        Raises:
            FileNotFoundError: If there is no file at ``path``.
            ImportError: If onnxruntime is not installed.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No exported policy at '{path}'.")

        try:
            import onnxruntime
        except ImportError as error:  # pragma: no cover - depends on the environment
            raise ImportError(
                "Running an exported policy needs onnxruntime. Install it with: pip install "
                "onnxruntime"
            ) from error

        self.path = path
        self.session = onnxruntime.InferenceSession(str(path))
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def obs_dim(self) -> int:
        """Number of observation values the policy expects."""
        return int(self.session.get_inputs()[0].shape[-1])

    @property
    def num_actions(self) -> int:
        """Number of action values the policy produces."""
        return int(self.session.get_outputs()[0].shape[-1])

    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Compute the action for an observation.

        The action is deterministic: the exported graph carries the mean of the trained policy,
        not a sample from it, which is what deployment wants.

        Args:
            obs: A single observation of shape ``(obs_dim,)`` or a batch of shape
                ``(batch, obs_dim)``.

        Returns:
            The action, shaped ``(num_actions,)`` for a single observation or
            ``(batch, num_actions)`` for a batch, already scaled to the environment's action range.
        """
        single = obs.ndim == 1
        batch = np.asarray(obs, dtype=np.float32)
        if single:
            batch = batch[None]

        action = self.session.run([self.output_name], {self.input_name: batch})[0]
        return action[0] if single else action

    def __repr__(self) -> str:
        """Return a short description of the loaded policy."""
        return f"Policy('{self.path.name}', obs_dim={self.obs_dim}, num_actions={self.num_actions})"
