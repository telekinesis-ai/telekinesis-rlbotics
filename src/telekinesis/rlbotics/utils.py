"""Shared helpers for resolving callables and devices."""

import importlib
import random
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger


def set_seed(seed: int) -> None:
    """Seed Python's, NumPy's and PyTorch's random number generators.

    Makes model initialization, action sampling and mini-batch shuffling reproducible from run to
    run. What this does not cover is the environment: initial states, domain randomization and any
    other simulator-side randomness are seeded by the adapter, if at all, which is a separate concern
    from the training algorithm's own randomness.

    Args:
        seed: Seed applied to every generator.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_callable(reference: Callable | str) -> Callable:
    """Resolve a callable given directly or named by an import path.

    Naming one by string is what lets a configuration be written to JSON and read back, which is how
    a run's ``config.json`` stays reproducible even when it points at user code.

    Args:
        reference: The callable itself, or a path to it as ``"module.path:attribute"`` or
            ``"module.path.attribute"``.

    Returns:
        The callable.

    Raises:
        TypeError: If the reference is neither callable nor a string.
        ImportError: If the module cannot be imported.
        AttributeError: If the module has no such attribute.
    """
    if callable(reference):
        return reference
    if not isinstance(reference, str):
        raise TypeError(
            f"Expected a callable or an import path to one, got {type(reference).__name__}."
        )

    module_path, separator, attribute = reference.rpartition(":" if ":" in reference else ".")
    if not separator:
        raise ImportError(
            f"'{reference}' is not an import path. Write it as 'module.path:function' or "
            "'module.path.function'."
        )

    resolved = importlib.import_module(module_path)
    for name in attribute.split("."):
        resolved = getattr(resolved, name)
    if not callable(resolved):
        raise TypeError(f"'{reference}' resolved to {type(resolved).__name__}, which is not callable.")
    return resolved


def resolve_device(requested: str = "auto") -> str:
    """Resolve the device to train on, falling back when the requested one is unavailable.

    Apple Silicon exposes its GPU as "mps". Note that it is not automatically faster here: MuJoCo
    steps on the CPU, so every rollout step pays a host-to-device copy, and these networks are small
    enough that the copies outweigh the faster matrix multiplies. Measured on an M-series Mac with
    the defaults below, CPU reached ~2400 steps/s against ~1200 on MPS. MPS pulls ahead once the
    networks and batches are large, so it is worth trying with wider layers or many environments.

    Resolving is idempotent: passing a value this function returned gives the same value back.
    That matters because the device is resolved in more than one place, so a resolved "cuda:0"
    must not be mistaken for an unknown device and quietly downgraded to cpu.

    Args:
        requested: "auto", or a torch device such as "cpu", "mps", "cuda" or "cuda:1". "auto"
            prefers mps, then cuda, then cpu.

    Returns:
        A device string that torch can use.
    """
    available = {
        "cpu": True,
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }

    if requested == "auto":
        for candidate in ("mps", "cuda", "cpu"):
            if available[candidate]:
                if candidate != "cpu":
                    logger.info(f"   Auto-selected '{candidate}'. Pass --device cpu to compare.")
                return "cuda:0" if candidate == "cuda" else candidate

    # A device may carry an index, as in "cuda:1", so availability is checked on the kind alone and
    # the index is preserved
    kind = requested.split(":")[0]
    if not available.get(kind, False):
        logger.warning(f"   Device '{requested}' is not available on this machine, using cpu.")
        return "cpu"
    return "cuda:0" if requested == "cuda" else requested
