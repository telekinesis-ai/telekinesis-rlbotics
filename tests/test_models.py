"""Tests for neural network models."""

import pytest
import torch
import torch.nn as nn
from telekinesis.rlbotics.config import CNNConfig, CNNEncoderConfig, GaussianDistributionConfig, MLPConfig
from telekinesis.rlbotics.models import MLP, CNN, MLPModel, CNNModel
from telekinesis.rlbotics.distributions import GaussianDistribution, SquashedGaussianDistribution


class TestMLP:
    """Test MLP module."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        mlp = MLP(input_dim=10, output_dim=5, hidden_dims=(32, 32))
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 5)

    def test_dynamic_hidden_dims(self):
        """Test -1 placeholder for hidden dimensions."""
        mlp = MLP(input_dim=10, output_dim=5, hidden_dims=(-1, 32))
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 5)

    def test_output_reshaping(self):
        """Test output reshaping."""
        mlp = MLP(input_dim=10, output_dim=(4, 3), hidden_dims=(32,))
        x = torch.randn(2, 10)
        y = mlp(x)
        assert y.shape == (2, 4, 3)

    def test_last_activation(self):
        """Test last activation function."""
        mlp = MLP(
            input_dim=10,
            output_dim=5,
            hidden_dims=(32,),
            last_activation="sigmoid",
        )
        x = torch.randn(4, 10)
        y = mlp(x)
        assert y.shape == (4, 5)
        assert (y >= 0).all() and (y <= 1).all()

    def test_different_activations(self):
        """Test different activation functions."""
        for activation in ["relu", "elu", "tanh", "sigmoid", "leaky_relu", "gelu"]:
            mlp = MLP(input_dim=10, output_dim=5, activation=activation)
            x = torch.randn(4, 10)
            y = mlp(x)
            assert y.shape == (4, 5)
            assert torch.isfinite(y).all()

    def test_weight_initialization(self):
        """Test orthogonal weight initialization."""
        mlp = MLP(input_dim=10, output_dim=5, hidden_dims=(32,))
        mlp.init_weights(gain=1.0)

        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                # Check that weights are not all zeros
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                # Check that biases are zero
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestCNN:
    """Test CNN module."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16, 32),
            kernel_size=3,
            stride=1,
            activation="elu",
            flatten=True,
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert y.shape[0] == 4
        assert len(y.shape) == 2  # Flattened

    def test_output_channels_not_flattened(self):
        """Test output_channels property when not flattened."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16, 32),
            flatten=False,
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert cnn.output_channels == 32
        assert len(y.shape) == 4

    def test_max_pooling(self):
        """Test max pooling."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16,),
            max_pool=True,
            flatten=True,
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert y.shape[0] == 4

    def test_global_pooling(self):
        """Test global average pooling."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16,),
            global_pool="avg",
            flatten=True,
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert y.shape == (4, 16)

    def test_batch_normalization(self):
        """Test batch normalization layers."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16, 32),
            norm="batch",
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert y.shape[0] == 4

    def test_activation_functions(self):
        """Test different activation functions."""
        for activation in ["relu", "elu", "tanh", "sigmoid", "gelu", "swish"]:
            cnn = CNN(
                input_dim=(32, 32),
                input_channels=3,
                output_channels=(16,),
                activation=activation,
            )
            x = torch.randn(4, 3, 32, 32)
            y = cnn(x)
            assert y.shape[0] == 4
            assert torch.isfinite(y).all()

    def test_per_layer_normalization(self):
        """Test per-layer normalization configuration."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16, 32),
            norm=("batch", "none"),
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert y.shape[0] == 4

    def test_per_layer_pooling(self):
        """Test per-layer max pooling configuration."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16, 32),
            max_pool=(True, False),
        )
        x = torch.randn(4, 3, 32, 32)
        y = cnn(x)
        assert y.shape[0] == 4

    def test_invalid_normalization(self):
        """Test error handling for invalid normalization."""
        with pytest.raises(ValueError):
            cnn = CNN(
                input_dim=(32, 32),
                input_channels=3,
                output_channels=(16,),
                norm="invalid",
            )
            x = torch.randn(4, 3, 32, 32)
            cnn(x)

    def test_invalid_global_pooling(self):
        """Test error handling for invalid global pooling."""
        with pytest.raises(ValueError):
            CNN(
                input_dim=(32, 32),
                input_channels=3,
                output_channels=(16,),
                global_pool="invalid",
            )

    def test_weight_initialization(self):
        """Test Kaiming weight initialization."""
        cnn = CNN(
            input_dim=(32, 32),
            input_channels=3,
            output_channels=(16, 32),
        )
        cnn.init_weights()

        for module in cnn.modules():
            if isinstance(module, nn.Conv2d):
                assert not torch.allclose(module.weight, torch.zeros_like(module.weight))
                assert torch.allclose(module.bias, torch.zeros_like(module.bias))


class TestMLPModel:
    """Test MLPModel."""

    def test_forward_deterministic(self):
        """Test deterministic forward pass."""
        model = MLPModel(MLPConfig(hidden_dims=(32, 32)), input_dim=10, output_dim=5)
        x = torch.randn(4, 10)
        y = model(x, stochastic=False)
        assert y.shape == (4, 5)

    def test_forward_with_distribution(self):
        """Test forward pass with distribution."""
        cfg = MLPConfig(
            hidden_dims=(32, 32),
            distribution_cfg=GaussianDistributionConfig(init_std=1.0),
        )
        model = MLPModel(cfg, input_dim=10, output_dim=5)

        x = torch.randn(4, 10)
        y = model(x, stochastic=True)
        assert y.shape == (4, 5)

    def test_get_latent(self):
        """Test latent extraction."""
        model = MLPModel(MLPConfig(hidden_dims=(32,)), input_dim=10, output_dim=5)
        x = torch.randn(4, 10)
        latent = model.get_latent(x)
        assert latent.shape == (4, 10)

    def test_obs_normalization(self):
        """Test observation normalization."""
        model = MLPModel(MLPConfig(obs_normalization=True), input_dim=10, output_dim=5)
        x = torch.randn(100, 10)
        model.update_normalization(x)

        # After normalization, latent should be approximately normalized
        normalized = model.get_latent(x[:1])
        assert torch.isfinite(normalized).all()

    def test_distribution_properties(self):
        """Test distribution properties."""
        cfg = MLPConfig(
            hidden_dims=(32,),
            distribution_cfg=GaussianDistributionConfig(class_name="GaussianDistribution"),
        )
        model = MLPModel(cfg, input_dim=10, output_dim=5)

        x = torch.randn(4, 10)
        model(x, stochastic=True)

        assert model.output_mean.shape == (4, 5)
        assert model.output_std.shape == (4, 5)
        assert model.output_entropy.shape == (4,)
        assert model.output_distribution_params[0].shape == (4, 5)

    def test_log_prob(self):
        """Test log probability computation."""
        cfg = MLPConfig(
            hidden_dims=(32,),
            distribution_cfg=GaussianDistributionConfig(class_name="GaussianDistribution"),
        )
        model = MLPModel(cfg, input_dim=10, output_dim=5)

        x = torch.randn(4, 10)
        model(x, stochastic=True)
        actions = torch.randn(4, 5)
        log_prob = model.get_output_log_prob(actions)
        assert log_prob.shape == (4,)
        assert torch.isfinite(log_prob).all()

    def test_jit_export(self):
        """Test JIT export."""
        model = MLPModel(MLPConfig(), input_dim=10, output_dim=5)
        jit_model = model.as_jit()

        x = torch.randn(4, 10)
        y_orig = model(x, stochastic=False)
        y_jit = jit_model(x)

        assert torch.allclose(y_orig, y_jit, atol=1e-5)

    def test_onnx_export(self):
        """Test ONNX export."""
        model = MLPModel(MLPConfig(), input_dim=10, output_dim=5)
        onnx_model = model.as_onnx()

        x = torch.randn(4, 10)
        y_orig = model(x, stochastic=False)
        y_onnx = onnx_model(x)

        assert torch.allclose(y_orig, y_onnx, atol=1e-5)


class TestCNNModel:
    """Test CNNModel."""

    def test_forward_deterministic(self):
        """Test deterministic forward pass."""
        encoder_cfg = CNNEncoderConfig(
            output_channels=(16, 32),
        )
        model = CNNModel(
            CNNConfig(hidden_dims=(32,), cnn_cfg=encoder_cfg),
            input_dim=(32, 32),
            input_channels=3,
            output_dim=5,
        )
        x = torch.randn(4, 3, 32, 32)
        y = model(x, stochastic=False)
        assert y.shape == (4, 5)

    def test_forward_with_distribution(self):
        """Test forward pass with distribution."""
        encoder_cfg = CNNEncoderConfig(
            output_channels=(16, 32),
        )
        cfg = CNNConfig(
            hidden_dims=(32,),
            cnn_cfg=encoder_cfg,
            distribution_cfg=GaussianDistributionConfig(
                class_name="SquashedGaussianDistribution", init_std=0.5
            ),
        )
        model = CNNModel(cfg, input_dim=(32, 32), input_channels=3, output_dim=5)

        x = torch.randn(4, 3, 32, 32)
        y = model(x, stochastic=True)
        assert y.shape == (4, 5)
        assert (y >= -1.0).all() and (y <= 1.0).all()

    def test_get_latent(self):
        """Test latent extraction from CNN."""
        encoder_cfg = CNNEncoderConfig(
            output_channels=(16,),
        )
        model = CNNModel(
            CNNConfig(cnn_cfg=encoder_cfg), input_dim=(32, 32), input_channels=3, output_dim=5
        )
        x = torch.randn(4, 3, 32, 32)
        latent = model.get_latent(x)
        assert len(latent.shape) == 2
        assert latent.shape[0] == 4

    def test_distribution_properties(self):
        """Test distribution properties."""
        encoder_cfg = CNNEncoderConfig(
            output_channels=(16,),
        )
        cfg = CNNConfig(cnn_cfg=encoder_cfg, distribution_cfg=GaussianDistributionConfig())
        model = CNNModel(cfg, input_dim=(32, 32), input_channels=3, output_dim=5)

        x = torch.randn(4, 3, 32, 32)
        model(x, stochastic=True)

        assert model.output_mean.shape == (4, 5)
        assert model.output_std.shape == (4, 5)

    def test_jit_export(self):
        """Test JIT export."""
        encoder_cfg = CNNEncoderConfig(
            output_channels=(16,),
        )
        model = CNNModel(
            CNNConfig(cnn_cfg=encoder_cfg), input_dim=(32, 32), input_channels=3, output_dim=5
        )
        jit_model = model.as_jit()

        x = torch.randn(4, 3, 32, 32)
        y_orig = model(x, stochastic=False)
        y_jit = jit_model(x)

        assert torch.allclose(y_orig, y_jit, atol=1e-5)

    def test_onnx_export(self):
        """Test ONNX export."""
        encoder_cfg = CNNEncoderConfig(
            output_channels=(16,),
        )
        model = CNNModel(
            CNNConfig(cnn_cfg=encoder_cfg), input_dim=(32, 32), input_channels=3, output_dim=5
        )
        onnx_model = model.as_onnx()

        x = torch.randn(4, 3, 32, 32)
        y_orig = model(x, stochastic=False)
        y_onnx = onnx_model(x)

        assert torch.allclose(y_orig, y_onnx, atol=1e-5)
