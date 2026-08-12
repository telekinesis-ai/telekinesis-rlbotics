"""Tests for the training configurations under configs/, and for building one in Python.

A broken file would otherwise only be found when someone tries to train with it, so every shipped
configuration is built here through the same call ``training_example.py`` makes. Nothing is simulated:
the files are validated on their own, so mjlab and Isaac Lab are covered without either installed.
"""

import sys
from pathlib import Path

import pytest
import yaml

from telekinesis.rlbotics.config import OnPolicyRunnerConfig

REPO = Path(__file__).parent.parent
CONFIG_DIR = REPO / "configs"
sys.path.insert(0, str(REPO / "examples"))

# The simulators that have configurations, which are also the directory names
FRAMEWORKS = ("gymnasium", "mjlab", "isaaclab")

# The observation group each simulator publishes for the actor
ACTOR_GROUPS = {"gymnasium": ["observation"], "mjlab": ["actor"], "isaaclab": ["policy"]}

# The Gymnasium tasks the README documents, which are the ones expected to carry a configuration
DOCUMENTED_GYMNASIUM = {
    "Pendulum-v1", "MountainCarContinuous-v0", "InvertedPendulum-v5", "InvertedDoublePendulum-v5",
    "Reacher-v5", "Swimmer-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "Pusher-v5",
    "Ant-v5", "Humanoid-v5",
}


def config_files() -> list[Path]:
    """Return every configuration, so the tests below are parameterized over them."""
    return sorted(CONFIG_DIR.glob("*/*.yaml"))


def config_id(path: Path) -> str:
    """Return a test id naming the framework and the task, since task names repeat across them."""
    return f"{path.parent.name}/{path.stem}"


def plain(value):
    """Return a value with every tuple turned into a list, for comparing across YAML.

    A config built in Python keeps the sequence type it was given, while one loaded from YAML always
    has lists, so ``hidden_dims=(64, 64)`` and ``[64, 64]`` describe the same two layers. Comparing
    through this asserts the values match without asserting the type does.

    Args:
        value: A value from :meth:`BaseConfig.to_dict`.

    Returns:
        The value with tuples replaced by lists, recursively.
    """
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


@pytest.mark.parametrize("path", config_files(), ids=config_id)
def test_config_builds_a_runner_config(path):
    """Test that a file builds a valid configuration, nested values reaching it intact."""
    data = yaml.safe_load(path.read_text())

    # A configuration describes a whole run: the task it trains, and the runner that trains it
    assert set(data) == {"env", "runner"}, f"unexpected keys: {sorted(set(data) - {'env', 'runner'})}"

    runner = data["runner"]
    cfg = OnPolicyRunnerConfig.from_dict(runner)

    # The actor needs a distribution to sample from; a critic predicts one value and has none
    assert cfg.actor.distribution_cfg is not None
    assert cfg.critic.distribution_cfg is None
    # The file's values, not a default quietly substituted for them
    assert cfg.num_steps_per_env == runner["num_steps_per_env"]
    assert cfg.num_learning_iterations == runner["num_learning_iterations"]
    assert cfg.algorithm.learning_rate == runner["algorithm"]["learning_rate"]
    assert cfg.actor.hidden_dims == runner["actor"]["hidden_dims"]


@pytest.mark.parametrize("path", config_files(), ids=config_id)
def test_env_block_matches_where_the_file_lives(path):
    """Test that the directory and the file name are the framework and the task, not just labels.

    ``training_example.py`` dispatches on ``env.framework`` and builds ``env.id``, so a file whose
    block disagrees with its own path would train something other than what it is named after.
    """
    env = yaml.safe_load(path.read_text())["env"]

    assert env["framework"] == path.parent.name
    assert env["id"] == path.stem
    assert env["num_envs"] > 0


@pytest.mark.parametrize("path", config_files(), ids=config_id)
def test_obs_groups_match_the_framework(path):
    """Test that each configuration names the observation group its simulator publishes."""
    data = yaml.safe_load(path.read_text())

    assert data["runner"]["obs_groups"]["actor"] == ACTOR_GROUPS[data["env"]["framework"]]


@pytest.mark.parametrize("framework", FRAMEWORKS)
def test_every_framework_has_configurations(framework):
    """Test that each supported simulator ships at least one task to train."""
    assert sorted((CONFIG_DIR / framework).glob("*.yaml")), f"no configurations for {framework}"


def test_the_documented_gymnasium_tasks_are_configured():
    """Test that the Gymnasium tasks the README lists all have a file of their own."""
    configured = {path.stem for path in (CONFIG_DIR / "gymnasium").glob("*.yaml")}

    assert DOCUMENTED_GYMNASIUM <= configured, f"missing: {sorted(DOCUMENTED_GYMNASIUM - configured)}"


class TestBuildingInPython:
    """Test configuration_example.py, which builds the same configuration as code."""

    def test_the_example_builds_a_valid_config(self):
        """Test that the explicitly constructed configuration is complete and validated."""
        from configuration_example import make_runner_config

        cfg = make_runner_config(obs_dim=3, log_dir="logs")

        assert isinstance(cfg, OnPolicyRunnerConfig)
        assert cfg.actor.distribution_cfg is not None
        assert cfg.critic.distribution_cfg is None

    def test_networks_are_sized_from_the_observation(self):
        """Test the one decision a static file cannot make: capacity from the observation size."""
        from configuration_example import make_runner_config

        small = make_runner_config(obs_dim=3, log_dir="logs")
        large = make_runner_config(obs_dim=348, log_dir="logs")

        assert small.actor.hidden_dims == (64, 64)
        assert large.actor.hidden_dims == (256, 256, 128)

    def test_a_written_config_round_trips(self, tmp_path):
        """Test that what the example writes is what the training example can read back.

        This is the claim the two scripts make together: build a configuration in Python, write it,
        and train with it as YAML.
        """
        from configuration_example import make_runner_config, write_config

        built = make_runner_config(obs_dim=3, log_dir=str(tmp_path))
        path = write_config(built, tmp_path / "generated.yaml", env_id="Pendulum-v1", num_envs=8)

        data = yaml.safe_load(path.read_text())
        assert set(data) == {"env", "runner"}
        assert data["env"] == {
            "framework": "gymnasium",
            "id": "Pendulum-v1",
            "num_envs": 8,
            "device": "auto",
        }

        reloaded = OnPolicyRunnerConfig.from_dict(data["runner"])

        assert plain(reloaded.to_dict()) == plain(built.to_dict())
        assert reloaded.actor.hidden_dims == list(built.actor.hidden_dims)
        assert reloaded.algorithm.learning_rate == built.algorithm.learning_rate

    def test_the_written_config_is_plain_yaml(self, tmp_path):
        """Test that no Python-specific tags leak in, which safe_load would refuse."""
        from configuration_example import make_runner_config, write_config

        built = make_runner_config(obs_dim=3, log_dir=str(tmp_path))
        path = write_config(built, tmp_path / "generated.yaml", env_id="Pendulum-v1", num_envs=8)

        assert "!!python" not in path.read_text()


class TestFrameworkDispatch:
    """Test that training_example.py routes a configuration to the right adapter."""

    def test_an_unknown_framework_is_rejected(self):
        """Test that a typo names what is supported rather than failing obscurely."""
        from training_example import make_env

        with pytest.raises(ValueError, match="Unknown env.framework 'nope'"):
            make_env({"framework": "nope", "id": "Whatever-v0", "num_envs": 1})

    @pytest.mark.parametrize("framework", FRAMEWORKS)
    def test_every_configured_framework_is_dispatched(self, framework):
        """Test that no shipped configuration names a framework the example cannot build."""
        from training_example import FRAMEWORKS as DISPATCHED

        assert framework in DISPATCHED
