"""Run every example in this directory tree and report which ones pass.

Examples are collected recursively, so the training scripts next to this file and the demos under
``module_examples/`` are all covered. Each runs in its own subprocess, so a failure in one does not
stop the others and every example gets a clean interpreter. The exit code is non-zero if any example
failed, which makes this usable as a smoke test::

    python examples/run_all_examples.py                  # run all
    python examples/run_all_examples.py mlp cnn          # only names containing 'mlp' or 'cnn'
    python examples/run_all_examples.py module_examples   # only the module demos

A filter matches against the path relative to this directory, so a directory name selects
everything inside it.
"""

import subprocess
import sys
import time
from pathlib import Path

from loguru import logger

EXAMPLES_DIR = Path(__file__).parent

# Examples that are not standalone demos and should not be run directly
EXCLUDED = {Path(__file__).name}

# An example whose optional dependency is missing exits with this code, so it is reported as skipped
# rather than as a failure. training_example.py does this when a configuration asks for a simulator
# that is not installed, which is expected for mjlab and isaaclab on most machines.
SKIPPED = 2

# Arguments for an example's default run, keyed by its path relative to this directory. Chosen to be
# a real but cheap training run: train, checkpoint, export, deploy.
EXAMPLE_ARGS = {
    "training_example.py": [
        "configs/gymnasium/Pendulum-v1.yaml", "-i", "5", "-d", "cpu",
    ],
    # Already a short run on Pendulum, so only the iteration count is worth cutting
    "configuration_example.py": ["-i", "3", "-d", "cpu"],
}

# training_example.py covers three simulators from one script by reading env.framework out of the
# configuration, so its default run above (Gymnasium) is not the whole story: it is run once more per
# entry here, against a task cheap enough to be a smoke test rather than a real training run.
EXTRA_RUNS = {
    "training_example.py": [
        ("mjlab", ["configs/mjlab/Mjlab-Cartpole-Balance.yaml", "-n", "8", "-i", "3", "-d", "cpu"]),
        ("isaaclab", ["configs/isaaclab/Isaac-Cartpole-v0.yaml", "-n", "64", "-i", "3", "-d", "cpu"]),
    ],
}


def label(path: Path) -> str:
    """Return the example's path relative to this directory, which is how it is named throughout.

    Args:
        path: Path to the example script.

    Returns:
        The relative path, such as "module_examples/mlp_example.py".
    """
    return path.relative_to(EXAMPLES_DIR).as_posix()


def find_examples(name_filters: list[str] | None = None) -> list[Path]:
    """Collect the example scripts to run, from this directory and every directory below it.

    Args:
        name_filters: Only keep examples whose relative path contains one of these substrings, so
            a directory name selects everything inside it. Defaults to None, which keeps them all.

    Returns:
        The example scripts, shallowest first, so the training scripts come before the module demos.
    """
    examples = sorted(
        (
            path
            for path in EXAMPLES_DIR.rglob("*.py")
            if path.name not in EXCLUDED
            and not path.name.startswith("_")
            and "__pycache__" not in path.parts
        ),
        key=lambda path: (len(path.relative_to(EXAMPLES_DIR).parts), label(path)),
    )
    if name_filters:
        examples = [path for path in examples if any(f in label(path) for f in name_filters)]
    return examples


def find_jobs(name_filters: list[str] | None = None) -> list[tuple[Path, list[str], str]]:
    """Expand the example scripts into the runs to execute, each with its own arguments and name.

    Most examples run once. One that covers more than one simulator, such as
    ``training_example.py``, gets one run per :data:`EXTRA_RUNS` entry in addition to its default,
    since a single pass at the default arguments would leave the other simulators untested.

    Args:
        name_filters: Forwarded to :func:`find_examples`.

    Returns:
        Triples of the script to run, the arguments to run it with, and the name to report it under.
    """
    jobs = []
    for path in find_examples(name_filters):
        name = label(path)
        jobs.append((path, EXAMPLE_ARGS.get(name, []), name))
        jobs.extend(
            (path, args, f"{name} ({variant})") for variant, args in EXTRA_RUNS.get(name, [])
        )
    return jobs


def run_example(path: Path, args: list[str]) -> tuple[str, float, str]:
    """Run a single example in a subprocess.

    Args:
        path: Path to the example script.
        args: Command-line arguments to run it with.

    Returns:
        Its status, one of "PASS", "SKIP" or "FAIL", how long it took in seconds, and its combined
        output.
    """
    start = time.time()
    result = subprocess.run(
        [sys.executable, str(path), *args],
        capture_output=True,
        text=True,
        cwd=EXAMPLES_DIR.parent,
    )
    duration = time.time() - start
    status = {0: "PASS", SKIPPED: "SKIP"}.get(result.returncode, "FAIL")
    return status, duration, result.stdout + result.stderr


def main(name_filters: list[str] | None = None) -> int:
    """Run the examples and print a summary.

    Args:
        name_filters: Only run examples whose file name contains one of these substrings.

    Returns:
        Process exit code: 0 if every example passed, 1 otherwise.
    """
    jobs = find_jobs(name_filters)
    if not jobs:
        logger.error(f"No examples matched {name_filters}")
        return 1

    logger.info(f"Running {len(jobs)} examples")
    logger.info("=" * 70)

    failures = []
    skipped = []
    for path, args, name in jobs:
        status, duration, output = run_example(path, args)
        if status == "PASS":
            logger.info(f"PASS  {name:<48} {duration:6.1f}s")
        elif status == "SKIP":
            logger.warning(f"SKIP  {name:<48} {duration:6.1f}s  optional dependency missing")
            skipped.append(name)
        else:
            logger.error(f"FAIL  {name:<48} {duration:6.1f}s")
            # Show the tail of the output, which is where the traceback is
            for line in output.strip().split("\n")[-15:]:
                logger.error(f"      {line}")
            failures.append(name)

    logger.info("=" * 70)
    if failures:
        logger.error(f"{len(failures)} of {len(jobs)} examples failed: {', '.join(failures)}")
        return 1

    passed = len(jobs) - len(skipped)
    if skipped:
        logger.info(f"All {passed} examples passed, {len(skipped)} skipped: {', '.join(skipped)}")
        return 0

    logger.info(f"All {passed} examples passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
