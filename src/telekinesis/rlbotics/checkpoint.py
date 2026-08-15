"""Checkpoint writing, rotation and resuming."""

from __future__ import annotations

import math
from pathlib import Path

import torch
from loguru import logger

from telekinesis.rlbotics.config import RESUME_CHOICES, LoggerConfig

CHECKPOINT_STEM = "model"
"""Prefix of a checkpoint file name, followed by the iteration: ``model_100.pt``."""

BEST_CHECKPOINT = f"{CHECKPOINT_STEM}_best.pt"
"""Name of the best-scoring checkpoint. Not iteration-numbered, so rotation leaves it alone."""


class CheckpointManager:
    """Writes checkpoints on a schedule, keeps the most recent ones, and finds one to resume from.

    Periodic checkpoints are named ``model_<iteration>.pt``, so they carry the iteration they belong
    to and can be ordered by it rather than by file name, where ``model_100`` would sort before
    ``model_25``. Alongside them the manager keeps ``model_best.pt``, rewritten whenever the metric
    it is given improves, which is what makes a good policy survive a run that later collapses.
    """

    def __init__(self, cfg: LoggerConfig, run_dir: str | None = None) -> None:
        """Initialize the manager.

        Args:
            cfg: Logging configuration, which carries the checkpoint settings.
            run_dir: Directory of the current run, where checkpoints are written. Defaults to None,
                which disables writing, though resuming from an earlier run still works.
        """
        self.cfg = cfg
        self.directory = Path(run_dir) if run_dir is not None else None
        self.best_metric: float | None = None
        """Best metric seen so far, or None if nothing has been scored yet.

        Seeded from the checkpoint when a run resumes, so a run that never reaches the previous
        best does not overwrite it with something worse.
        """

    @property
    def enabled(self) -> bool:
        """Whether checkpoints are being written."""
        return self.cfg.enabled and self.directory is not None

    def should_save(self, iteration: int) -> bool:
        """Return whether a checkpoint is due at this iteration.

        Args:
            iteration: The iteration that just finished.

        Returns:
            True if a checkpoint should be written.
        """
        return self.enabled and iteration % self.cfg.save_interval == 0

    def save(self, state: dict, iteration: int) -> Path | None:
        """Write a checkpoint and drop any that fall outside ``keep_last_n``.

        Args:
            state: The state dictionary to save.
            iteration: Iteration the checkpoint belongs to, used in the file name.

        Returns:
            The path written, or None if checkpointing is disabled.
        """
        if not self.enabled:
            return None

        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{CHECKPOINT_STEM}_{iteration}.pt"
        torch.save(state, path)
        self._prune()
        return path

    def save_best(self, state: dict, metric: float, iteration: int) -> Path | None:
        """Write ``model_best.pt`` if this is the best metric seen, and return the path written.

        The metric is maximized, so it should be something like the mean episode reward. Called every
        iteration: a best checkpoint that only landed on the ``save_interval`` grid would miss the
        peak of a reward curve that later collapses.

        Args:
            state: The state dictionary to save. The metric and iteration are recorded in it, so the
                file says what it scored and when.
            metric: Value to compare against the best so far. Higher is better.
            iteration: Iteration the checkpoint belongs to.

        Returns:
            The path written, or None if checkpointing is off, ``save_best`` is off, or the metric did
            not improve.
        """
        if not self.enabled or not self.cfg.save_best:
            return None
        # A NaN compares False against everything, so without this guard it would read as an
        # improvement, become the score to beat, and let the next worse policy overwrite the best
        if not math.isfinite(metric):
            logger.warning(f"   Ignoring a non-finite score ({metric}) for the best checkpoint.")
            return None
        if self.best_metric is not None and metric <= self.best_metric:
            return None

        self.best_metric = metric
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / BEST_CHECKPOINT
        torch.save({**state, "best_metric": metric, "best_iteration": iteration}, path)
        return path

    def checkpoints(self, directory: str | Path | None = None) -> list[Path]:
        """Return the checkpoints in a directory, oldest iteration first.

        Args:
            directory: Directory to list. Defaults to None, meaning this run's directory.

        Returns:
            The checkpoints, ordered by the iteration in their name.
        """
        directory = Path(directory) if directory is not None else self.directory
        if directory is None or not directory.is_dir():
            return []
        found = []
        for path in directory.glob(f"{CHECKPOINT_STEM}_*.pt"):
            iteration = self._iteration_of(path)
            # A name with no iteration in it is not one of ours, so leave it alone
            if iteration is not None:
                found.append((iteration, path))
        return [path for _, path in sorted(found)]

    @staticmethod
    def _iteration_of(path: Path) -> int | None:
        """Return the iteration a checkpoint's name carries, or None if it carries none.

        Args:
            path: Path to a checkpoint.

        Returns:
            The iteration, or None for a file that is not a periodic checkpoint of ours, which
            includes the best checkpoint since its name records no iteration.
        """
        try:
            return int(path.stem.removeprefix(f"{CHECKPOINT_STEM}_"))
        except ValueError:
            return None

    def latest(self) -> Path | None:
        """Return the checkpoint from the highest iteration, or None if there is none."""
        found = self.checkpoints()
        return found[-1] if found else None

    def best(self) -> Path | None:
        """Return this run's best checkpoint, or None if none has been written."""
        if self.directory is None:
            return None
        path = self.directory / BEST_CHECKPOINT
        return path if path.is_file() else None

    def resume_path(self) -> Path | None:
        """Return the checkpoint to resume from, as ``cfg.resume`` asks for.

        ``None`` starts from scratch. ``"last"`` and ``"best"`` search every run of the experiment,
        so training continues from wherever it left off rather than from this run's empty directory.
        Anything else names a checkpoint: an existing path is taken as given, otherwise it is treated
        as a file name to look for among the experiment's runs.

        Returns:
            The checkpoint to load, or None if training should start from scratch.

        Raises:
            FileNotFoundError: If a resume was requested but no matching checkpoint was found.
        """
        wanted = self.cfg.resume
        if wanted is None:
            return None

        if wanted not in RESUME_CHOICES:
            path = Path(wanted)
            # A path that exists needs no searching, and works with no log_dir set at all
            if path.is_file():
                return path
            # Anything carrying a directory was meant as a path, so say that rather than going on to
            # search the experiment for a file of that name
            if path.parent != Path("."):
                raise FileNotFoundError(f"No checkpoint at '{wanted}' to resume from.")

        if self.cfg.experiment_dir is None:
            raise FileNotFoundError(
                f"Resuming from '{wanted}' was requested but 'log_dir' is None, so there is nowhere "
                "to look for a checkpoint. Set 'log_dir', or give a path to a checkpoint file."
            )
        experiment_dir = Path(self.cfg.experiment_dir)

        if wanted == "best":
            best = self._best_of_experiment(experiment_dir)
            if best is not None:
                return best
            logger.warning(
                f"   No '{BEST_CHECKPOINT}' under '{experiment_dir}', resuming from the most "
                "recent checkpoint instead."
            )
            wanted = "last"

        if wanted == "last":
            # The best checkpoint carries no iteration, so it is not what "last" means
            candidates = [
                path
                for path in experiment_dir.rglob(f"{CHECKPOINT_STEM}_*.pt")
                if path.name != BEST_CHECKPOINT
            ]
        else:
            candidates = list(experiment_dir.rglob(Path(wanted).name))

        if not candidates:
            raise FileNotFoundError(
                f"Resuming from '{self.cfg.resume}' was requested but no matching checkpoint was "
                f"found under '{experiment_dir}'. Train once without resuming to produce one, or "
                "give a path to a checkpoint file."
            )
        # Two checkpoints written in quick succession can share an mtime to the nanosecond, since a
        # file's timestamp comes from a coarse clock that only advances every few milliseconds. The
        # iteration in the name breaks that tie, so "last" stays the later checkpoint rather than
        # whichever one the directory happens to list first. A name carrying no iteration sorts
        # first, leaving those ties to the order they were found in as before.
        def recency(path: Path) -> tuple[int, int]:
            iteration = self._iteration_of(path)
            return (path.stat().st_mtime_ns, -1 if iteration is None else iteration)

        return max(candidates, key=recency)

    @staticmethod
    def _best_of_experiment(experiment_dir: Path) -> Path | None:
        """Return the highest-scoring checkpoint across every run of an experiment.

        Each run keeps its own ``model_best.pt``, so the best of the experiment is the one whose
        recorded metric is highest, not the newest file.

        Args:
            experiment_dir: Directory holding the experiment's runs.

        Returns:
            The best checkpoint, or None if no run wrote one.
        """
        scored: list[tuple[float, Path]] = []
        for path in experiment_dir.rglob(BEST_CHECKPOINT):
            state = torch.load(path, weights_only=False, map_location="cpu")
            metric = state.get("best_metric")
            if metric is not None:
                scored.append((float(metric), path))

        if not scored:
            return None
        return max(scored, key=lambda scored_path: scored_path[0])[1]

    def _prune(self) -> None:
        """Delete the checkpoints that fall outside ``keep_last_n``.

        Files sharing a checkpoint's name are removed with it, so an animation recorded next to
        ``model_100.pt`` as ``model_100.gif`` does not outlive the checkpoint it illustrates.
        """
        if self.cfg.keep_last_n <= 0:
            return

        for path in self.checkpoints()[: -self.cfg.keep_last_n]:
            for sibling in path.parent.glob(f"{path.stem}.*"):
                sibling.unlink(missing_ok=True)
