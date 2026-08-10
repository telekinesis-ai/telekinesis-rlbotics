"""Tests for checkpoint writing, rotation and resuming."""

import pytest
import torch

from telekinesis.rlbotics.checkpoint import CheckpointManager
from telekinesis.rlbotics.config import LoggerConfig


def write(manager: CheckpointManager, iterations: list[int], with_gif: bool = False) -> None:
    """Write a checkpoint per iteration, optionally with a sibling animation."""
    for iteration in iterations:
        path = manager.save({"iteration": iteration}, iteration)
        if with_gif and path is not None:
            path.with_suffix(".gif").write_bytes(b"gif")


class TestRunDirectory:
    """Test where checkpoints are written."""

    def test_checkpoints_go_flat_into_the_run_directory(self, tmp_path):
        """Checkpoints sit beside the event file, not in a subdirectory."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))

        path = manager.save({"a": 1}, 0)

        assert path == tmp_path / "model_0.pt"

    def test_no_run_directory_disables_writing(self, tmp_path):
        """Without a run directory there is nowhere to write, so nothing is written."""
        manager = CheckpointManager(LoggerConfig())

        assert not manager.enabled
        assert manager.save({"a": 1}, 0) is None

    def test_disabled_writes_nothing(self, tmp_path):
        """A disabled config saves nothing."""
        manager = CheckpointManager(LoggerConfig(enabled=False), run_dir=str(tmp_path))

        assert manager.save({"a": 1}, 0) is None
        assert not list(tmp_path.glob("*.pt"))


class TestSaveInterval:
    """Test the checkpoint schedule."""

    def test_should_save_follows_the_interval(self, tmp_path):
        """Checkpoints are due on multiples of save_interval."""
        manager = CheckpointManager(LoggerConfig(save_interval=10), run_dir=str(tmp_path))

        due = [manager.should_save(i) for i in (0, 5, 10, 15, 20)]

        assert due == [True, False, True, False, True]

    def test_disabled_is_never_due(self, tmp_path):
        """A disabled manager never reports a checkpoint as due."""
        manager = CheckpointManager(
            LoggerConfig(enabled=False, save_interval=1), run_dir=str(tmp_path)
        )

        assert not manager.should_save(0)


class TestRotation:
    """Test that only the most recent checkpoints are kept."""

    def test_keeps_the_last_n(self, tmp_path):
        """Older checkpoints are deleted once keep_last_n is exceeded."""
        manager = CheckpointManager(LoggerConfig(keep_last_n=3), run_dir=str(tmp_path))

        write(manager, [0, 10, 20, 30, 40, 50])

        assert [p.name for p in manager.checkpoints()] == [
            "model_30.pt", "model_40.pt", "model_50.pt"
        ]

    def test_rotation_takes_sibling_files_with_it(self, tmp_path):
        """An animation recorded next to a checkpoint is removed with it."""
        manager = CheckpointManager(LoggerConfig(keep_last_n=2), run_dir=str(tmp_path))

        write(manager, [0, 10, 20], with_gif=True)

        assert sorted(p.name for p in manager.directory.glob("*.gif")) == [
            "model_10.gif", "model_20.gif"
        ]

    def test_keep_last_n_zero_keeps_everything(self, tmp_path):
        """A keep_last_n of zero disables rotation."""
        manager = CheckpointManager(LoggerConfig(keep_last_n=0), run_dir=str(tmp_path))

        write(manager, [0, 10, 20, 30])

        assert len(manager.checkpoints()) == 4


class TestOrdering:
    """Test that checkpoints are ordered by iteration, not by name."""

    def test_latest_uses_the_iteration_number(self, tmp_path):
        """model_100 is newer than model_25, even though it sorts earlier by name."""
        manager = CheckpointManager(LoggerConfig(keep_last_n=0), run_dir=str(tmp_path))

        write(manager, [25, 100])

        assert manager.latest().name == "model_100.pt"

    def test_unrelated_files_are_ignored(self, tmp_path):
        """A file that is not named model_<iteration>.pt is left out."""
        manager = CheckpointManager(LoggerConfig(keep_last_n=0), run_dir=str(tmp_path))
        write(manager, [0])
        (manager.directory / "model_best.pt").write_bytes(b"not ours")

        assert [p.name for p in manager.checkpoints()] == ["model_0.pt"]

    def test_latest_is_none_when_empty(self, tmp_path):
        """An empty directory has no latest checkpoint."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))

        assert manager.latest() is None


class TestResume:
    """Test resolving which checkpoint to resume from."""

    def test_no_resume_returns_none(self, tmp_path):
        """Training starts from scratch unless resuming was asked for."""
        manager = CheckpointManager(LoggerConfig(log_dir=str(tmp_path)), run_dir=str(tmp_path))
        write(manager, [0])

        assert manager.resume_path() is None

    def test_resume_finds_the_latest_across_runs(self, tmp_path):
        """Resuming looks across the experiment, so an earlier run's checkpoint is found."""
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp", keep_last_n=0)
        first_run = tmp_path / "exp" / "run_one"
        write(CheckpointManager(cfg, run_dir=str(first_run)), [0, 10])

        # A new run starts with an empty directory of its own
        second_run = tmp_path / "exp" / "run_two"
        manager = CheckpointManager(cfg.replace(resume="last"), run_dir=str(second_run))

        assert manager.checkpoints() == []
        assert manager.resume_path().name == "model_10.pt"

    def test_checkpoint_path_implies_resume(self, tmp_path):
        """Naming a checkpoint is enough, resume does not have to be set as well."""
        writer = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))
        write(writer, [0])
        target = writer.directory / "model_0.pt"

        manager = CheckpointManager(
            LoggerConfig(resume=str(target)), run_dir=str(tmp_path)
        )

        assert manager.cfg.resume
        assert manager.resume_path() == target

    def test_missing_checkpoint_path_raises(self, tmp_path):
        """A path that does not exist fails as a path, rather than being searched for by name."""
        manager = CheckpointManager(
            LoggerConfig(resume=str(tmp_path / "nope.pt")), run_dir=str(tmp_path)
        )

        with pytest.raises(FileNotFoundError, match="No checkpoint at"):
            manager.resume_path()

    def test_resume_with_nothing_to_resume_raises(self, tmp_path):
        """Asking to resume an empty experiment fails loudly rather than starting over silently."""
        manager = CheckpointManager(
            LoggerConfig(log_dir=str(tmp_path), experiment="empty", resume="last"),
            run_dir=str(tmp_path / "empty" / "run"),
        )

        with pytest.raises(FileNotFoundError, match="no matching checkpoint was found"):
            manager.resume_path()


class TestRoundTrip:
    """Test that a saved checkpoint reads back."""

    def test_saved_state_is_recoverable(self, tmp_path):
        """What was saved is what loads."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))
        state = {"iteration": 7, "weights": torch.ones(3)}

        path = manager.save(state, 7)
        loaded = torch.load(path, weights_only=False)

        assert loaded["iteration"] == 7
        assert torch.equal(loaded["weights"], torch.ones(3))


class TestBestCheckpoint:
    """Test keeping the best-scoring checkpoint, which the manager scores on a maximized metric."""

    def test_first_score_writes_the_best(self, tmp_path):
        """The first scored iteration is the best so far, so it lands as model_best.pt."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))

        path = manager.save_best({"a": 1}, metric=-500.0, iteration=3)

        assert path == tmp_path / "model_best.pt"
        assert manager.best_metric == -500.0

    def test_only_an_improvement_is_written(self, tmp_path):
        """A worse or equal score leaves the existing best in place."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))
        manager.save_best({"iteration": 1}, metric=100.0, iteration=1)

        assert manager.save_best({"iteration": 2}, metric=50.0, iteration=2) is None
        assert manager.save_best({"iteration": 3}, metric=100.0, iteration=3) is None
        assert manager.save_best({"iteration": 4}, metric=101.0, iteration=4) is not None

        state = torch.load(tmp_path / "model_best.pt", weights_only=False)
        assert state["iteration"] == 4
        assert state["best_metric"] == 101.0
        assert state["best_iteration"] == 4

    def test_records_what_it_scored(self, tmp_path):
        """The file says what it scored and when, so it can be compared across runs."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))
        manager.save_best({"a": 1}, metric=42.5, iteration=7)

        state = torch.load(tmp_path / "model_best.pt", weights_only=False)
        assert state["best_metric"] == 42.5
        assert state["best_iteration"] == 7

    def test_survives_rotation(self, tmp_path):
        """Rotation drops old periodic checkpoints but never the best one."""
        manager = CheckpointManager(LoggerConfig(keep_last_n=2), run_dir=str(tmp_path))
        manager.save_best({"a": 1}, metric=10.0, iteration=0)
        write(manager, [0, 1, 2, 3, 4])

        assert (tmp_path / "model_best.pt").is_file()
        assert [path.name for path in manager.checkpoints()] == ["model_3.pt", "model_4.pt"]

    def test_can_be_turned_off(self, tmp_path):
        """With save_best off nothing is written, whatever the score."""
        manager = CheckpointManager(LoggerConfig(save_best=False), run_dir=str(tmp_path))

        assert manager.save_best({"a": 1}, metric=1000.0, iteration=1) is None
        assert not (tmp_path / "model_best.pt").exists()

    def test_best_reports_this_run(self, tmp_path):
        """best() returns the path once written, and None before that."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))

        assert manager.best() is None
        manager.save_best({"a": 1}, metric=1.0, iteration=1)
        assert manager.best() == tmp_path / "model_best.pt"


class TestResumeFrom:
    """Test which checkpoint a resume starts from."""

    def _experiment(self, tmp_path, first_metric: float, second_metric: float):
        """Write two runs of one experiment, each with a periodic and a best checkpoint."""
        experiment = tmp_path / "exp"
        for index, metric in enumerate((first_metric, second_metric)):
            cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp")
            manager = CheckpointManager(cfg, run_dir=str(experiment / f"run_{index}"))
            manager.save({"iteration": index}, index)
            manager.save_best({"iteration": index}, metric=metric, iteration=index)
        return experiment

    def test_last_is_the_default(self, tmp_path):
        """Without asking, a resume continues from the most recent checkpoint."""
        self._experiment(tmp_path, first_metric=900.0, second_metric=100.0)
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp", resume="last")

        resumed = CheckpointManager(cfg).resume_path()

        assert resumed.parent.name == "run_1"
        assert resumed.name == "model_1.pt"

    def test_best_picks_the_highest_score_across_runs(self, tmp_path):
        """resume_from="best" compares the recorded metrics, not the file times.

        The better policy is in the older run here, so picking the newest file would lose it.
        """
        self._experiment(tmp_path, first_metric=900.0, second_metric=100.0)
        cfg = LoggerConfig(
            log_dir=str(tmp_path), experiment="exp", resume="best"
        )

        resumed = CheckpointManager(cfg).resume_path()

        assert resumed.parent.name == "run_0"
        assert resumed.name == "model_best.pt"
        assert torch.load(resumed, weights_only=False)["best_metric"] == 900.0

    def test_best_falls_back_to_last(self, tmp_path):
        """An experiment with no best checkpoint yet still resumes, from the most recent one."""
        experiment = tmp_path / "exp"
        manager = CheckpointManager(
            LoggerConfig(log_dir=str(tmp_path), experiment="exp"),
            run_dir=str(experiment / "run_0"),
        )
        manager.save({"iteration": 5}, 5)
        cfg = LoggerConfig(
            log_dir=str(tmp_path), experiment="exp", resume="best"
        )

        assert CheckpointManager(cfg).resume_path().name == "model_5.pt"

    def test_explicit_path_overrides_the_choice(self, tmp_path):
        """A named checkpoint wins over resume_from."""
        experiment = self._experiment(tmp_path, first_metric=900.0, second_metric=100.0)
        wanted = experiment / "run_1" / "model_1.pt"
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp", resume=str(wanted))

        assert CheckpointManager(cfg).resume_path() == wanted

    def test_last_ignores_the_best_file(self, tmp_path):
        """The best checkpoint is not a candidate for "last", whose name carries no iteration."""
        experiment = tmp_path / "exp"
        manager = CheckpointManager(
            LoggerConfig(log_dir=str(tmp_path), experiment="exp"),
            run_dir=str(experiment / "run_0"),
        )
        manager.save({"iteration": 1}, 1)
        # Written after the periodic checkpoint, so it is the newest file on disk
        manager.save_best({"iteration": 1}, metric=1.0, iteration=1)
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp", resume="last")

        assert CheckpointManager(cfg).resume_path().name == "model_1.pt"

    def test_a_named_checkpoint_is_looked_up_in_the_experiment(self, tmp_path):
        """A bare file name is searched for among the experiment's runs, no path needed."""
        experiment = self._experiment(tmp_path, first_metric=900.0, second_metric=100.0)
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp", resume="model_0.pt")

        resumed = CheckpointManager(cfg).resume_path()

        assert resumed == experiment / "run_0" / "model_0.pt"

    def test_an_unknown_name_says_what_it_looked_for(self, tmp_path):
        """A name that matches nothing fails with the name and the directory searched."""
        self._experiment(tmp_path, first_metric=1.0, second_metric=2.0)
        cfg = LoggerConfig(log_dir=str(tmp_path), experiment="exp", resume="model_999.pt")

        with pytest.raises(FileNotFoundError, match="model_999.pt"):
            CheckpointManager(cfg).resume_path()

    def test_a_full_path_is_taken_as_given(self, tmp_path):
        """An existing path is used directly, so it works without log_dir set."""
        manager = CheckpointManager(LoggerConfig(), run_dir=str(tmp_path))
        wanted = manager.save({"iteration": 3}, 3)
        cfg = LoggerConfig(log_dir=None, resume=str(wanted))

        assert CheckpointManager(cfg).resume_path() == wanted
