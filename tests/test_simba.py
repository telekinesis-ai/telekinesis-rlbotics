"""Tests for SimBaModel, its config, custom model class resolution, and environment log metrics.

None of these need AssembleNIST installed: the model and the runner wiring are exercised against a
mock environment that publishes the same two observation groups (points, then proprioception).
"""

import pytest
import torch
from tensordict import TensorDict

from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    LoggerConfig,
    MLPConfig,
    OnPolicyRunnerConfig,
    PPOConfig,
    SimBaConfig,
    _model_config_from_dict,
)
from telekinesis.rlbotics.envs.base import VecEnv
from telekinesis.rlbotics.logger import Logger
from telekinesis.rlbotics.models import MLPModel, SimBaModel
from telekinesis.rlbotics.runner import OnPolicyRunner

POINT_DIM = 12
PROPRIO_DIM = 5


class CustomSimBa(SimBaModel):
    """A user-defined SimBa variant, named by import path in the config tests."""


class NotAModel:
    """An importable class that extends none of the registered models."""


def small_simba(**overrides) -> SimBaConfig:
    """A tiny SimBa config, quick to build and run."""
    kwargs = dict(point_dim=POINT_DIM, encoder_dims=(8, 4), hidden_dim=8, expansion=16, num_blocks=1)
    kwargs.update(overrides)
    return SimBaConfig(**kwargs)


class PointsEnv(VecEnv):
    """Mock environment with a ``points`` and a ``proprio`` observation group."""

    def __init__(self, num_envs: int = 4, num_actions: int = 3) -> None:
        self.num_envs = num_envs
        self.num_actions = num_actions
        self.device = torch.device("cpu")
        self.max_episode_length = 10
        self.cfg = {}
        self._t = 0

    def _obs(self) -> TensorDict:
        return TensorDict(
            {"points": torch.randn(self.num_envs, POINT_DIM), "proprio": torch.randn(self.num_envs, PROPRIO_DIM)},
            batch_size=(self.num_envs,),
        )

    def reset(self) -> TensorDict:
        return self._obs()

    def get_observations(self) -> TensorDict:
        return self._obs()

    def step(self, actions):
        self._t += 1
        dones = torch.zeros(self.num_envs, dtype=torch.bool)
        if self._t % 3 == 0:
            dones[0] = True
        extras = {"time_outs": torch.zeros(self.num_envs, dtype=torch.bool), "log": {"success_rate": 0.5}}
        return self._obs(), torch.randn(self.num_envs), dones, extras


class TestSimBaConfig:
    def test_factory_routes_simba_by_class_name(self):
        cfg = _model_config_from_dict({"class_name": "SimBaModel", "point_dim": 30})
        assert isinstance(cfg, SimBaConfig)
        assert cfg.point_dim == 30

    def test_factory_still_builds_plain_mlp_configs(self):
        assert type(_model_config_from_dict({"class_name": "MLPModel"})) is MLPConfig

    def test_import_path_to_a_custom_subclass_gets_its_parents_config(self):
        """Test that a custom model named by import path is configured as the model it extends."""
        cfg = _model_config_from_dict({"class_name": f"{__name__}:CustomSimBa", "point_dim": 30})

        assert type(cfg) is SimBaConfig
        assert cfg.class_name == f"{__name__}:CustomSimBa"

    def test_custom_subclass_passed_directly_gets_its_parents_config(self):
        """Test that a subclass passed as the class itself is not mistaken for a plain MLP."""
        assert type(_model_config_from_dict({"class_name": CustomSimBa, "point_dim": 30})) is SimBaConfig

    def test_qualified_built_in_name_is_accepted(self):
        """Test that a built-in model can also be named by its full import path."""
        cfg = _model_config_from_dict({"class_name": "telekinesis.rlbotics.models:SimBaModel"})
        assert type(cfg) is SimBaConfig

    @pytest.mark.parametrize(
        "class_name,match",
        [
            ("SimbaModel", "not one of"),                          # a typo is not silently an MLP
            ("my_missing_pkg.models:Model", "not one of"),         # cannot be imported
            (f"{__name__}:NotAModel", "derives from none of"),     # importable, but no known base
        ],
    )
    def test_unknown_model_fails_instead_of_falling_back(self, class_name, match):
        """Test that a model no config fits is an error at load time, not an MLPConfig."""
        with pytest.raises(ValueError, match=match):
            _model_config_from_dict({"class_name": class_name})

    def test_runner_config_rejects_a_non_model_config(self):
        """Test that a wrong config object for the actor fails when the config is built, not in the runner."""
        with pytest.raises(ValueError, match="'actor' must be a model config"):
            OnPolicyRunnerConfig(obs_groups={"actor": ["policy"], "critic": ["policy"]}, actor=PPOConfig())

    def test_cnn_cfg_without_a_cnn_model_is_rejected(self):
        """Test that a CNN encoder config is not silently dropped or used to guess the model."""
        with pytest.raises(ValueError, match="cnn_cfg"):
            _model_config_from_dict({"cnn_cfg": {}})
        with pytest.raises(ValueError, match="cnn_cfg"):
            _model_config_from_dict({"class_name": "SimBaModel", "cnn_cfg": {}})

    def test_every_model_has_exactly_one_config_and_one_constructor(self):
        """Test that config parsing and model building agree on the same model names.

        Configs are picked by the names in MODEL_CONFIGS and models by the keys of MODELS, so a
        model added to one and not the other would parse into the wrong config or fail to build.
        """
        from telekinesis.rlbotics.config import MODEL_CONFIGS
        from telekinesis.rlbotics.models import MODELS

        configured = [name for cls in MODEL_CONFIGS for name in cls.expected_class_names]
        assert sorted(configured) == sorted(MODELS)
        for name, model_class in MODELS.items():
            assert callable(getattr(model_class, "from_obs", None)), name
            assert type(_model_config_from_dict({"class_name": name})).expected_class_names == (name,)

    @pytest.mark.parametrize(
        "field,value", [("point_dim", 0), ("num_blocks", 0), ("encoder_dims", []), ("head_gain", 0.0)]
    )
    def test_rejects_invalid_settings(self, field, value):
        with pytest.raises(ValueError):
            small_simba(**{field: value})


class TestSimBaModel:
    def test_actor_output_shape_and_distribution(self):
        cfg = small_simba(distribution_cfg=GaussianDistributionConfig())
        model = SimBaModel(cfg, input_dim=POINT_DIM + PROPRIO_DIM, output_dim=3)
        x = torch.randn(7, POINT_DIM + PROPRIO_DIM)

        actions = model(x)
        assert actions.shape == (7, 3)
        assert model.get_output_log_prob(actions).shape == (7,)
        assert model(x, stochastic=False).shape == (7, 3)

    def test_is_a_drop_in_mlp_model(self):
        assert issubclass(SimBaModel, MLPModel)

    def test_input_without_proprio_is_rejected(self):
        with pytest.raises(ValueError, match="point inputs"):
            SimBaModel(small_simba(), input_dim=POINT_DIM, output_dim=1)

    def test_normalization_touches_only_proprio(self):
        model = SimBaModel(small_simba(obs_normalization=True), input_dim=POINT_DIM + PROPRIO_DIM, output_dim=1)
        model.train()
        x = torch.randn(64, POINT_DIM + PROPRIO_DIM) * 5 + 3
        model.update_normalization(x)

        y = model.get_latent(x)
        assert torch.equal(y[:, :POINT_DIM], x[:, :POINT_DIM])
        assert y[:, POINT_DIM:].mean().abs() < 0.5

    def test_head_gain_sets_small_initial_actions(self):
        model = SimBaModel(small_simba(head_gain=0.01), input_dim=POINT_DIM + PROPRIO_DIM, output_dim=3)
        assert model.mlp.head.weight.abs().max() < 0.05

    def test_onnx_wrapper_matches_deterministic_output(self):
        cfg = small_simba(obs_normalization=True, distribution_cfg=GaussianDistributionConfig())
        model = SimBaModel(cfg, input_dim=POINT_DIM + PROPRIO_DIM, output_dim=3)
        model.update_normalization(torch.randn(32, POINT_DIM + PROPRIO_DIM))
        model.eval()
        x = torch.randn(4, POINT_DIM + PROPRIO_DIM)

        assert torch.allclose(model.as_onnx()(x), model(x, stochastic=False), atol=1e-6)


class TestRunnerWiring:
    def make_cfg(self, actor_class="SimBaModel") -> OnPolicyRunnerConfig:
        actor = small_simba(class_name=actor_class, distribution_cfg=GaussianDistributionConfig())
        return OnPolicyRunnerConfig(
            obs_groups={"actor": ["points", "proprio"], "critic": ["points", "proprio"]},
            num_steps_per_env=4,
            logger=LoggerConfig(log_dir=None, log_interval=1),
            algorithm=PPOConfig(num_learning_epochs=1, num_mini_batches=1),
            actor=actor,
            critic=small_simba(),
            verbose=False,
        )

    def test_runner_builds_the_named_model_class(self):
        runner = OnPolicyRunner(PointsEnv(), self.make_cfg(), device="cpu")
        assert isinstance(runner.alg.actor, SimBaModel)
        assert isinstance(runner.alg.critic, SimBaModel)

    def test_runner_resolves_an_import_path(self):
        runner = OnPolicyRunner(PointsEnv(), self.make_cfg("telekinesis.rlbotics.models:SimBaModel"), device="cpu")
        assert isinstance(runner.alg.actor, SimBaModel)

    def test_learns_a_few_iterations(self):
        runner = OnPolicyRunner(PointsEnv(), self.make_cfg(), device="cpu")
        runner.learn(num_learning_iterations=2)
        assert runner.current_learning_iteration == 1


class TestEnvironmentLogMetrics:
    def test_log_extras_become_custom_metrics_and_are_cleared(self, capsys):
        logger = Logger(LoggerConfig(log_dir=None), num_envs=2, device="cpu")
        logger.process_env_step(torch.zeros(2), torch.zeros(2), {"success_rate": 0.25})
        logger.process_env_step(torch.zeros(2), torch.zeros(2), {"success_rate": torch.tensor(0.75), "other": 1})
        logger.process_env_step(torch.zeros(2), torch.zeros(2), {"vector": torch.ones(3)})

        assert logger._mean_episode_infos() == {"success_rate": 0.5, "other": 1.0}
        logger.log(iteration=0, total_iterations=1, print_interval=1)
        assert "success_rate" in capsys.readouterr().out
        assert logger.episode_info_buffer == []
