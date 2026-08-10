"""Tests for PPO algorithm."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict


from telekinesis.rlbotics.algorithms import PPO
from telekinesis.rlbotics.symmetry import Symmetry
from telekinesis.rlbotics.config import (
    GaussianDistributionConfig,
    MLPConfig,
    PPOConfig,
    SymmetryConfig,
)
from telekinesis.rlbotics.models import MLPModel
from telekinesis.rlbotics.rollout import RolloutBuffer


def make_mlp(input_dim: int, output_dim: int, hidden_dim: int = 64) -> nn.Sequential:
    """Create a simple MLP."""
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class TestPPOInitialization:
    """Test PPO initialization."""

    def test_initialization(self):
        """Test basic PPO initialization."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(),
            actor=actor,
            critic=critic,
            storage=storage,
            device="cpu",
        )

        assert ppo.actor is not None
        assert ppo.critic is not None
        assert ppo.device == "cpu"

    def test_hyperparameters(self):
        """Test PPO hyperparameter initialization."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(learning_rate=1e-4, clip_param=0.3, gamma=0.98, lam=0.92),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        assert ppo.learning_rate == 1e-4
        assert ppo.cfg.clip_param == 0.3
        assert ppo.cfg.gamma == 0.98
        assert ppo.cfg.lam == 0.92

    def test_get_learning_rate(self):
        """Test getting learning rate."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(learning_rate=1e-4),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        lr = ppo.get_learning_rate()
        assert lr == 1e-4

    def test_get_action_std(self):
        """Test getting action std."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        std = ppo.get_action_std()
        assert std is None


class TestPPOModes:
    """Test PPO mode switching."""

    def test_train_mode(self):
        """Test train mode."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        ppo.train_mode()
        assert ppo.actor.training
        assert ppo.critic.training

    def test_eval_mode(self):
        """Test eval mode."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        ppo.eval_mode()
        assert not ppo.actor.training
        assert not ppo.critic.training


class TestPPOUtilities:
    """Test PPO utility methods."""

    def test_get_policy(self):
        """Test getting policy network."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        policy = ppo.get_policy()
        assert policy is ppo._raw_actor

    def test_save_and_load(self):
        """Test saving and loading models."""
        actor = make_mlp(8, 2)
        critic = make_mlp(8, 1)
        storage = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")

        ppo = PPO(
            cfg=PPOConfig(),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        # Save state
        saved_dict = ppo.save()
        assert "actor_state_dict" in saved_dict
        assert "critic_state_dict" in saved_dict
        assert "optimizer_state_dict" in saved_dict

        # Create new PPO and load
        actor2 = make_mlp(8, 2)
        critic2 = make_mlp(8, 1)
        storage2 = RolloutBuffer("rl", num_envs=4, num_transitions_per_env=5, obs_shape=(8,), actions_shape=(2,), device="cpu")
        ppo2 = PPO(
            cfg=PPOConfig(),
            actor=actor2,
            critic=critic2,
            storage=storage2,
        )

        ppo2.load(saved_dict, load_cfg=None, strict=True)


class TestTimeoutBootstrapping:
    """Test the value bootstrap applied to episodes cut off by the time limit.

    :meth:`~telekinesis.rlbotics.env.VecEnv.step` documents rewards and dones as ``(num_envs,)``,
    so these tests pin that contract: the bootstrap arithmetic used to assume ``(num_envs, 1)`` and
    raised a broadcasting error for the documented shape.
    """

    @staticmethod
    def _ppo(num_envs: int = 4, obs_dim: int = 6, act_dim: int = 3, gamma: float = 0.99) -> PPO:
        """Build a PPO instance with a stochastic actor over a small observation."""
        actor = MLPModel(
            MLPConfig(hidden_dims=(8,), distribution_cfg=GaussianDistributionConfig()),
            input_dim=obs_dim,
            output_dim=act_dim,
        )
        critic = MLPModel(MLPConfig(hidden_dims=(8,)), input_dim=obs_dim, output_dim=1)
        storage = RolloutBuffer(
            "rl", num_envs, 2, (obs_dim,), (act_dim,), "cpu"
        )
        return PPO(cfg=PPOConfig(gamma=gamma), actor=actor, critic=critic, storage=storage)

    def _step_once(self, ppo: PPO, timed_out: bool) -> torch.Tensor:
        """Run one act/process cycle with the documented shapes and return the stored rewards."""
        num_envs, obs_dim = ppo.storage.num_envs, ppo.storage.obs_shape[0]
        obs = TensorDict({"observation": torch.randn(num_envs, obs_dim)}, batch_size=(num_envs,))
        ppo.act(obs)
        # Pin the value estimate so the expected bootstrap is exact
        ppo.transition.values = torch.full((num_envs, 1), 5.0)
        rewards = torch.ones(num_envs)
        dones = torch.full((num_envs,), float(timed_out))
        time_outs = torch.full((num_envs,), timed_out, dtype=torch.bool)
        ppo.process_env_step(obs, rewards, dones, {"time_outs": time_outs})
        return ppo.storage.rewards[0]

    def test_timeout_adds_discounted_value(self):
        """A timed-out step gets gamma * V(s) added to its reward."""
        ppo = self._ppo(gamma=0.99)

        stored = self._step_once(ppo, timed_out=True)

        # reward 1.0 + gamma 0.99 * value 5.0
        assert torch.allclose(stored, torch.full_like(stored, 5.95), atol=1e-5)

    def test_no_timeout_leaves_reward_untouched(self):
        """A step that did not time out keeps its reward."""
        ppo = self._ppo()

        stored = self._step_once(ppo, timed_out=False)

        assert torch.allclose(stored, torch.ones_like(stored), atol=1e-6)

    def test_gamma_scales_the_bootstrap(self):
        """The bootstrap is discounted by gamma."""
        stored = self._step_once(self._ppo(gamma=0.5), timed_out=True)

        # reward 1.0 + gamma 0.5 * value 5.0
        assert torch.allclose(stored, torch.full_like(stored, 3.5), atol=1e-5)

    def test_missing_time_outs_is_allowed(self):
        """An environment that reports no time_outs key skips the bootstrap."""
        ppo = self._ppo()
        num_envs, obs_dim = ppo.storage.num_envs, ppo.storage.obs_shape[0]
        obs = TensorDict({"observation": torch.randn(num_envs, obs_dim)}, batch_size=(num_envs,))
        ppo.act(obs)
        ppo.transition.values = torch.full((num_envs, 1), 5.0)

        ppo.process_env_step(obs, torch.ones(num_envs), torch.zeros(num_envs), {})

        assert torch.allclose(ppo.storage.rewards[0], torch.ones(num_envs, 1), atol=1e-6)


def mirror(env=None, obs=None, actions=None):
    """A stand-in mirror function: negating is a symmetry of a task that is symmetric about zero.

    Follows the contract the extension expects: each returned tensor is the originals stacked with
    their mirrored copies, and either argument may be None.
    """
    mirrored_obs = None if obs is None else torch.cat([obs, -obs], dim=0)
    mirrored_actions = None if actions is None else torch.cat([actions, -actions], dim=0)
    return mirrored_obs, mirrored_actions


class TestSymmetry:
    """Test the symmetry extension: data augmentation and the mirror loss."""

    def _batch(self, batch_size: int = 6, obs_dim: int = 4, num_actions: int = 2):
        """Build a minimal mini-batch of the shape PPO's update sees."""
        return SimpleNamespace(
            observations=torch.randn(batch_size, obs_dim),
            critic_observations=None,
            actions=torch.randn(batch_size, num_actions),
            old_actions_log_prob=torch.randn(batch_size, 1),
            values=torch.randn(batch_size, 1),
            advantages=torch.randn(batch_size, 1),
            returns=torch.randn(batch_size, 1),
        )

    def test_augmentation_appends_the_mirrored_samples(self):
        """Every rollout tensor grows with the observations, or the losses would not line up."""
        symmetry = Symmetry(SymmetryConfig(mirror, use_mirror_loss=False))
        batch = self._batch(batch_size=6)

        symmetry.augment_batch(batch, original_batch_size=6)

        assert batch.observations.shape == (12, 4)
        assert batch.actions.shape == (12, 2)
        for name in ("old_actions_log_prob", "values", "advantages", "returns"):
            assert getattr(batch, name).shape == (12, 1), name
        # The originals come first, so slicing to the original size still selects them
        assert torch.allclose(batch.observations[6:], -batch.observations[:6])

    def test_a_separate_critic_observation_set_is_augmented_too(self):
        """An asymmetric critic has to grow with the actor, or its value loss shapes mismatch."""
        symmetry = Symmetry(SymmetryConfig(mirror, use_mirror_loss=False))
        batch = self._batch(batch_size=6)
        batch.critic_observations = torch.randn(6, 9)

        symmetry.augment_batch(batch, original_batch_size=6)

        assert batch.critic_observations.shape == (12, 9)

    def test_augmentation_is_skipped_when_off(self):
        """With data augmentation off the batch is left exactly as it was."""
        symmetry = Symmetry(SymmetryConfig(mirror, use_data_augmentation=False))
        batch = self._batch(batch_size=6)

        symmetry.augment_batch(batch, original_batch_size=6)

        assert batch.observations.shape == (6, 4)

    def test_mirror_loss_is_zero_for_a_symmetric_policy(self):
        """A policy that is already odd about zero has nothing to gain, so the loss vanishes."""
        symmetry = Symmetry(SymmetryConfig(mirror, use_data_augmentation=False))
        batch = self._batch(batch_size=6)

        # An odd function: f(-x) = -f(x), which is exactly the symmetry being asked for
        def odd_actor(obs, stochastic=True):
            return obs[:, :2] * 2.0

        loss = symmetry.compute_loss(odd_actor, batch, original_batch_size=6)

        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mirror_loss_is_positive_for_an_asymmetric_policy(self):
        """A policy with a constant bias breaks the symmetry, and the loss says so."""
        symmetry = Symmetry(SymmetryConfig(mirror, use_data_augmentation=False))
        batch = self._batch(batch_size=6)

        def biased_actor(obs, stochastic=True):
            return obs[:, :2] + 5.0

        loss = symmetry.compute_loss(biased_actor, batch, original_batch_size=6)

        assert loss.item() > 1.0

    def test_the_loss_is_detached_when_it_is_only_reported(self):
        """With both flags off the loss is measured but must not reach the gradients."""
        reported = Symmetry(
            SymmetryConfig(mirror, use_data_augmentation=False, use_mirror_loss=False)
        )
        used = Symmetry(SymmetryConfig(mirror, use_data_augmentation=False))
        weight = torch.nn.Linear(4, 2)

        def actor(obs, stochastic=True):
            return weight(obs)

        assert not reported.compute_loss(actor, self._batch(), 6).requires_grad
        assert used.compute_loss(actor, self._batch(), 6).requires_grad

    def test_the_mirror_function_can_be_named_by_import_path(self):
        """A config that has to survive a JSON round-trip names the function instead of passing it.

        Compared by behaviour rather than identity: pytest imports this module under its own name, so
        resolving the path yields an equivalent function object, not the same one.
        """
        symmetry = Symmetry(SymmetryConfig("tests.test_ppo:mirror"))

        assert symmetry.data_augmentation_func.__name__ == "mirror"
        mirrored, _ = symmetry.data_augmentation_func(obs=torch.ones(2, 3), actions=None)
        assert mirrored.shape == (4, 3)

    def _trained_ppo(self, symmetry: SymmetryConfig | None = None) -> PPO:
        """Build a PPO over a small task and fill one rollout, ready to update."""
        num_envs, obs_dim, act_dim, steps = 4, 4, 2, 3
        actor = MLPModel(
            MLPConfig(hidden_dims=(8,), distribution_cfg=GaussianDistributionConfig()),
            input_dim=obs_dim,
            output_dim=act_dim,
        )
        critic = MLPModel(MLPConfig(hidden_dims=(8,)), input_dim=obs_dim, output_dim=1)
        storage = RolloutBuffer("rl", num_envs, steps, (obs_dim,), (act_dim,), "cpu")
        alg = PPO(
            cfg=PPOConfig(num_learning_epochs=1, num_mini_batches=1, symmetry_cfg=symmetry),
            actor=actor,
            critic=critic,
            storage=storage,
        )

        for _ in range(steps):
            obs = TensorDict(
                {"observation": torch.randn(num_envs, obs_dim)}, batch_size=(num_envs,)
            )
            alg.act(obs)
            alg.process_env_step(obs, torch.randn(num_envs), torch.zeros(num_envs), {})
        alg.compute_returns(obs)
        return alg

    def test_ppo_trains_with_symmetry(self):
        """End to end: PPO updates with augmentation and the mirror loss, and reports the loss."""
        alg = self._trained_ppo(SymmetryConfig(mirror, mirror_loss_coeff=0.5))

        loss_dict = alg.update()

        assert "symmetry" in loss_dict
        assert loss_dict["symmetry"] >= 0.0
        assert all(torch.isfinite(p).all() for p in alg.actor.parameters())
