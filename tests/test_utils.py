"""Tests for the shared utilities."""

import random
from unittest.mock import patch

import numpy as np
import torch

from telekinesis.rlbotics.utils import resolve_device, set_seed


class TestSetSeed:
    """Test the RNG-seeding helper that reproducible runs are built on."""

    def test_same_seed_reproduces_torchs_draw(self):
        """Test that reseeding puts torch's generator back in the same state."""
        set_seed(7)
        first = torch.randn(4)
        set_seed(7)
        second = torch.randn(4)

        assert torch.equal(first, second)

    def test_different_seeds_diverge(self):
        """Test that two different seeds are not (for all practical purposes) the same draw."""
        set_seed(7)
        first = torch.randn(4)
        set_seed(8)
        second = torch.randn(4)

        assert not torch.equal(first, second)

    def test_numpy_and_the_stdlib_generator_are_seeded_too(self):
        """Test that NumPy and Python's own random module move together with torch's.

        Nothing in this library samples from them directly today, but a user's environment or reward
        function easily could, and a "seed" that only covered torch would silently miss that.
        """
        set_seed(7)
        first = (random.random(), np.random.rand())
        set_seed(7)
        second = (random.random(), np.random.rand())

        assert first == second


def with_devices(mps: bool, cuda: bool):
    """Patch device availability so the resolution can be tested off-hardware.

    Args:
        mps: Whether MPS should report as available.
        cuda: Whether CUDA should report as available.

    Returns:
        A context manager applying both patches.
    """
    return _Patches(mps, cuda)


class _Patches:
    """Context manager patching MPS and CUDA availability together."""

    def __init__(self, mps: bool, cuda: bool) -> None:
        """Record the availability to report."""
        self.patches = [
            patch.object(torch.backends.mps, "is_available", return_value=mps),
            patch.object(torch.cuda, "is_available", return_value=cuda),
        ]

    def __enter__(self):
        """Apply the patches."""
        for item in self.patches:
            item.start()
        return self

    def __exit__(self, *exc_info) -> None:
        """Undo the patches."""
        for item in self.patches:
            item.stop()


class TestAutoSelection:
    """Test what "auto" picks."""

    def test_prefers_mps(self):
        """MPS wins when it is available."""
        with with_devices(mps=True, cuda=False):
            assert resolve_device("auto") == "mps"

    def test_falls_back_to_cuda(self):
        """CUDA is used when MPS is unavailable, with an explicit device index."""
        with with_devices(mps=False, cuda=True):
            assert resolve_device("auto") == "cuda:0"

    def test_falls_back_to_cpu(self):
        """CPU is the last resort."""
        with with_devices(mps=False, cuda=False):
            assert resolve_device("auto") == "cpu"


class TestExplicitDevice:
    """Test explicitly requested devices."""

    def test_available_device_is_kept(self):
        """An available device is returned unchanged."""
        with with_devices(mps=True, cuda=False):
            assert resolve_device("mps") == "mps"
            assert resolve_device("cpu") == "cpu"

    def test_cuda_gains_an_index(self):
        """A bare "cuda" is resolved to the first device."""
        with with_devices(mps=False, cuda=True):
            assert resolve_device("cuda") == "cuda:0"

    def test_device_index_is_preserved(self):
        """A specific GPU stays selected rather than collapsing to the first one or to cpu."""
        with with_devices(mps=False, cuda=True):
            assert resolve_device("cuda:1") == "cuda:1"

    def test_unavailable_device_falls_back_to_cpu(self):
        """Asking for hardware that is not there degrades to cpu instead of failing."""
        with with_devices(mps=False, cuda=False):
            assert resolve_device("mps") == "cpu"
            assert resolve_device("cuda") == "cpu"

    def test_unknown_device_falls_back_to_cpu(self):
        """An unrecognized name degrades to cpu."""
        with with_devices(mps=True, cuda=True):
            assert resolve_device("tpu") == "cpu"


class TestIdempotence:
    """Test that resolving twice is the same as resolving once.

    The device is resolved in more than one place: the training script may resolve "auto", and both
    the environment and the runner resolve whatever they are given. A resolved value must therefore
    survive being passed back in. It used to not: "cuda:0" was not a key in the availability table,
    so it was treated as unavailable and silently downgraded to cpu on a GPU machine.
    """

    def test_resolved_value_survives_a_second_pass(self):
        """Feeding the result back in returns the same device."""
        for mps, cuda in [(True, False), (False, True), (False, False)]:
            with with_devices(mps=mps, cuda=cuda):
                once = resolve_device("auto")
                assert resolve_device(once) == once
                assert resolve_device(resolve_device(once)) == once

    def test_cuda_is_not_downgraded_on_a_gpu_machine(self):
        """The whole chain a training script goes through keeps the GPU."""
        with with_devices(mps=False, cuda=True):
            script = resolve_device("auto")
            env = resolve_device(script)
            runner = resolve_device(script)

            assert (script, env, runner) == ("cuda:0", "cuda:0", "cuda:0")
