"""MLP and CNN models for actors and critics."""

from __future__ import annotations

import copy
import math
import torch
import torch.nn as nn
from functools import reduce

from telekinesis.rlbotics.config import CNNConfig, CNNEncoderConfig, GaussianDistributionConfig, MLPConfig
from telekinesis.rlbotics.distributions import Distribution, GaussianDistribution, SquashedGaussianDistribution
from telekinesis.rlbotics.normalization import EmpiricalNormalization

# Distribution classes that can be named by a distribution config
DISTRIBUTIONS = {
    "GaussianDistribution": GaussianDistribution,
    "SquashedGaussianDistribution": SquashedGaussianDistribution,
}


def build_distribution(cfg: GaussianDistributionConfig, act_dim: int) -> Distribution:
    """Build a distribution module from its config.

    Args:
        cfg: Distribution configuration.
        act_dim: Dimension of the distribution output.

    Returns:
        The configured distribution module.

    Raises:
        ValueError: If the configured distribution class is unknown.
    """
    # A class passed directly is used as-is, a name is looked up
    if callable(cfg.class_name):
        distribution_class = cfg.class_name
    elif cfg.class_name in DISTRIBUTIONS:
        distribution_class = DISTRIBUTIONS[cfg.class_name]
    else:
        raise ValueError(
            f"Unknown distribution '{cfg.class_name}'. Supported: {list(DISTRIBUTIONS)}, or pass the class directly."
        )

    distribution = distribution_class(act_dim=act_dim, init_std=cfg.init_std, std_range=cfg.std_range)
    # A standard deviation that is not learned is a frozen parameter
    distribution.log_std_param.requires_grad_(cfg.learn_std)
    return distribution


def _get_param(param, idx):
    """Get a parameter for the given index.

    Args:
        param: Parameter or list/tuple of parameters.
        idx: Index to get the parameter for.
    """
    if isinstance(param, (tuple, list)):
        return param[idx]
    return param


def _resolve_activation(act_name: str) -> nn.Module:
    """Resolve activation function from name.

    Args:
        act_name: Name of the activation function.

    Returns:
        The activation function module.
    """
    activations = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "leaky_relu": nn.LeakyReLU(),
        "gelu": nn.GELU(),
        "swish": nn.SiLU(),
        "identity": nn.Identity(),
    }
    if act_name not in activations:
        raise ValueError(
            f"Unknown activation: {act_name}. Supported: {list(activations.keys())}"
        )
    return activations[act_name]


def _compute_padding(
    input_hw: tuple[int, int],
    kernel: int,
    stride: int,
    dilation: int,
) -> tuple[int, int]:
    """Compute optimal padding for convolution layer.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    h = math.ceil(
        (stride * math.floor(input_hw[0] / stride) - input_hw[0]
         - stride + dilation * (kernel - 1) + 1) / 2
    )
    w = math.ceil(
        (stride * math.floor(input_hw[1] / stride) - input_hw[1]
         - stride + dilation * (kernel - 1) + 1) / 2
    )
    return (h, w)


def _compute_output_dim(
    input_hw: tuple[int, int],
    kernel: int,
    stride: int,
    dilation: int,
    padding: tuple[int, int],
    is_max_pool: bool = False,
) -> tuple[int, int]:
    """Compute output spatial dimensions after convolution.

    Reference: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
    """
    h = math.floor(
        (input_hw[0] + 2 * padding[0] - dilation * (kernel - 1) - 1) / stride + 1
    )
    w = math.floor(
        (input_hw[1] + 2 * padding[1] - dilation * (kernel - 1) - 1) / stride + 1
    )

    if is_max_pool:
        h = math.ceil(h / 2)
        w = math.ceil(w / 2)

    return (h, w)


class MLP(nn.Sequential):
    """Multi-Layer Perceptron.

    The MLP is a sequence of linear layers and activation functions. Features:
    - Dynamic hidden dimensions (supports -1 to infer from input)
    - Orthogonal weight initialization
    - Optional last activation function
    - Optional output reshaping
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple[int, ...] | list[int],
        hidden_dims: tuple[int, ...] | list[int] = (256, 256),
        activation: str = "relu",
        last_activation: str | None = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dim: Dimension of the input.
            output_dim: Dimension of the output. Can be tuple for reshaping.
            hidden_dims: Dimensions of hidden layers. -1 infers from input_dim.
            activation: Activation function (relu, elu, tanh, sigmoid).
            last_activation: Optional activation for the last layer.
        """
        super().__init__()

        # Resolve activation functions
        activation_mod = _resolve_activation(activation)
        last_activation_mod = (
            _resolve_activation(last_activation)
            if last_activation is not None
            else None
        )

        # Resolve hidden dimensions (replace -1 with input_dim)
        hidden_dims_resolved = [
            input_dim if dim == -1 else dim for dim in hidden_dims
        ]

        # Build layers
        layers = []

        # First layer: input -> first hidden
        layers.append(nn.Linear(input_dim, hidden_dims_resolved[0]))
        layers.append(activation_mod)

        # Hidden layers
        for i in range(len(hidden_dims_resolved) - 1):
            layers.append(
                nn.Linear(hidden_dims_resolved[i], hidden_dims_resolved[i + 1])
            )
            layers.append(activation_mod)

        # Output layer
        if isinstance(output_dim, int):
            layers.append(nn.Linear(hidden_dims_resolved[-1], output_dim))
        else:
            total_out_dim = reduce(lambda x, y: x * y, output_dim)
            layers.append(nn.Linear(hidden_dims_resolved[-1], total_out_dim))
            layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

        # Last activation if specified
        if last_activation_mod is not None:
            layers.append(last_activation_mod)

        # Register all layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    def init_weights(self, gain: float = 1.0) -> None:
        """Initialize weights using orthogonal initialization.

        Args:
            gain: Gain for orthogonal initialization.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gain)
                nn.init.zeros_(module.bias)


class CNN(nn.Sequential):
    """Convolutional Neural Network.

    A sequence of convolutional layers with optional normalization, activation, and pooling.
    Features:
    - Flexible layer configuration
    - Dilation support
    - Normalization support (batch, layer, none)
    - Optional max pooling and global pooling
    - Optional flattening
    - Kaiming weight initialization
    """

    def __init__(
        self,
        input_dim: tuple[int, int],
        input_channels: int,
        output_channels: tuple[int, ...] | list[int],
        kernel_size: int | tuple[int, ...] | list[int] = 3,
        stride: int | tuple[int, ...] | list[int] = 1,
        dilation: int | tuple[int, ...] | list[int] = 1,
        padding: str = "none",
        norm: str | tuple[str, ...] | list[str] = "none",
        activation: str = "elu",
        max_pool: bool | tuple[bool, ...] | list[bool] = False,
        global_pool: str = "none",
        flatten: bool = True,
    ) -> None:
        """Initialize the CNN.

        Args:
            input_dim: (height, width) of the input.
            input_channels: Number of input channels.
            output_channels: List of output channels for each conv layer.
            kernel_size: Kernel size(s) for conv layers.
            stride: Stride(s) for conv layers.
            dilation: Dilation(s) for conv layers.
            padding: Padding type ('none', 'zeros', 'reflect', 'replicate', 'circular').
            norm: Normalization type ('none', 'batch', 'layer') for each layer.
            activation: Activation function to use after each layer.
            max_pool: Whether to apply max pooling after each layer.
            global_pool: Global pooling type ('none', 'max', 'avg').
            flatten: Whether to flatten the output.
        """
        super().__init__()

        activation_fn = _resolve_activation(activation)
        layers = []

        last_channels = input_channels
        last_dim = input_dim

        for idx in range(len(output_channels)):
            k = _get_param(kernel_size, idx)
            s = _get_param(stride, idx)
            d = _get_param(dilation, idx)
            n = _get_param(norm, idx)
            pool = _get_param(max_pool, idx)

            # Compute padding
            if padding in ["zeros", "reflect", "replicate", "circular"]:
                p = _compute_padding(last_dim, k, s, d)
            else:
                p = (0, 0)

            # Convolutional layer
            layers.append(
                nn.Conv2d(
                    in_channels=last_channels,
                    out_channels=output_channels[idx],
                    kernel_size=k,
                    stride=s,
                    padding=p,
                    dilation=d,
                    padding_mode=padding if padding in ["zeros", "reflect", "replicate", "circular"] else "zeros",
                )
            )

            # Normalization layer
            if n == "none":
                pass
            elif n == "batch":
                layers.append(nn.BatchNorm2d(output_channels[idx]))
            elif n == "layer":
                norm_dim = _compute_output_dim(last_dim, k, s, d, p)
                layers.append(nn.LayerNorm([output_channels[idx], norm_dim[0], norm_dim[1]]))
            else:
                raise ValueError(
                    f"Unsupported normalization type: {n}. Supported types are 'none', 'batch', and 'layer'."
                )

            # Activation function
            layers.append(activation_fn)

            # Max pooling
            if pool:
                layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
                last_dim = _compute_output_dim(last_dim, k, s, d, p, is_max_pool=True)
            else:
                last_dim = _compute_output_dim(last_dim, k, s, d, p)

            last_channels = output_channels[idx]

        # Global pooling
        if global_pool == "none":
            pass
        elif global_pool == "max":
            layers.append(nn.AdaptiveMaxPool2d((1, 1)))
            last_dim = (1, 1)
        elif global_pool == "avg":
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            last_dim = (1, 1)
        else:
            raise ValueError(
                f"Unsupported global pooling type: {global_pool}. Supported types are 'none', 'max', and 'avg'."
            )

        # Flattening
        if flatten:
            layers.append(nn.Flatten(start_dim=1))
            self._output_channels = None
            self._output_dim = last_channels * last_dim[0] * last_dim[1]
        else:
            self._output_channels = last_channels
            self._output_dim = last_dim

        # Register layers
        for idx, layer in enumerate(layers):
            self.add_module(f"{idx}", layer)

    @property
    def output_channels(self) -> int | None:
        """Get the number of output channels (None if flattened)."""
        return self._output_channels

    @property
    def output_dim(self) -> tuple[int, int] | int:
        """Get the output dimension (spatial or flattened)."""
        return self._output_dim

    def init_weights(self) -> None:
        """Initialize weights with Kaiming normal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                nn.init.zeros_(module.bias)


class MLPModel(nn.Module):
    """MLP-based model for policy or value functions.

    Combines an MLP with optional distribution for stochastic policies.
    Features:
    - Observation normalization
    - Distribution integration
    - Export to JIT and ONNX
    """

    def __init__(
        self,
        cfg: MLPConfig,
        input_dim: int,
        output_dim: int,
    ) -> None:
        """Initialize MLP model from its config.

        The hidden layers, activation, observation normalization, and output distribution come from the config, while
        the input and output dimensions come from the environment.

        Args:
            cfg: Model configuration.
            input_dim: Input dimension, i.e. the dimension of the observation set the model consumes.
            output_dim: Output dimension, e.g. the number of actions for an actor or 1 for a critic.
        """
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build the output distribution, if the model is stochastic
        self.distribution = (
            build_distribution(cfg.distribution_cfg, output_dim) if cfg.distribution_cfg is not None else None
        )

        # Observation normalization
        if cfg.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(input_dim)
        else:
            self.obs_normalizer = nn.Identity()

        # Determine MLP output dimension
        mlp_output_dim = (
            self.distribution.input_dim if self.distribution else output_dim
        )

        # Create MLP
        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=mlp_output_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
        )

        # Initialize distribution weights if provided
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self, x: torch.Tensor, stochastic: bool = True
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            stochastic: If True, sample from distribution (if available).

        Returns:
            Output tensor.
        """
        latent = self.get_latent(x)
        mlp_out = self.mlp(latent)

        if self.distribution is None:
            return mlp_out

        if stochastic:
            self.distribution.update(mlp_out)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_out)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation.

        Args:
            x: Input tensor.

        Returns:
            Normalized latent tensor.
        """
        return self.obs_normalizer(x)

    def update_normalization(self, x: torch.Tensor) -> None:
        """Update observation normalization statistics.

        Args:
            x: Batch of observations.
        """
        if isinstance(self.obs_normalizer, EmpiricalNormalization):
            self.obs_normalizer.update(x)

    @property
    def output_mean(self) -> torch.Tensor:
        """Get mean of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Get std of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Get entropy of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Get raw parameters of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log probability of outputs."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.log_prob(outputs)

    def as_jit(self) -> nn.Module:
        """Export model for TorchScript/JIT compilation."""
        return _TorchMLPModel(self)

    def as_onnx(self) -> nn.Module:
        """Export model for ONNX."""
        return _OnnxMLPModel(self)


class CNNModel(nn.Module):
    """CNN-based model for vision-based policies or value functions.

    Combines CNN feature extraction with MLP output head and optional distribution.
    """

    def __init__(
        self,
        cfg: CNNConfig,
        input_dim: tuple[int, int],
        input_channels: int,
        output_dim: int,
        obs_group: str | None = None,
    ) -> None:
        """Initialize CNN model from its config.

        The encoder, MLP head, and output distribution come from the config, while the image shape and the output
        dimension come from the environment.

        Args:
            cfg: Model configuration. Its ``cnn_cfg`` holds the encoder settings.
            input_dim: Spatial (height, width) of the input images.
            input_channels: Number of input image channels.
            output_dim: Output dimension, e.g. the number of actions for an actor or 1 for a critic.
            obs_group: Observation group this model consumes, used to select the encoder when ``cfg.cnn_cfg`` maps
                several groups to their own encoders. Defaults to None.

        Raises:
            ValueError: If the config has no encoder settings, or none matching ``obs_group``.
        """
        super().__init__()
        self.cfg = cfg
        self.input_dim = input_dim
        self.input_channels = input_channels
        self.output_dim = output_dim

        # Select the encoder config, which may be a single config or one per observation group
        encoder_cfg = cfg.cnn_cfg
        if isinstance(encoder_cfg, dict):
            if obs_group is not None:
                if obs_group not in encoder_cfg:
                    raise ValueError(
                        f"No CNN encoder is configured for observation group '{obs_group}'. Configured groups:"
                        f" {sorted(encoder_cfg)}."
                    )
                encoder_cfg = encoder_cfg[obs_group]
            elif len(encoder_cfg) == 1:
                encoder_cfg = next(iter(encoder_cfg.values()))
            else:
                raise ValueError(
                    f"'cnn_cfg' configures several encoders ({sorted(encoder_cfg)}), so 'obs_group' is required to"
                    f" select one."
                )
        if not isinstance(encoder_cfg, CNNEncoderConfig):
            raise ValueError("A CNN model requires 'cnn_cfg' to be set on its config.")

        # Build the output distribution, if the model is stochastic
        self.distribution = (
            build_distribution(cfg.distribution_cfg, output_dim) if cfg.distribution_cfg is not None else None
        )

        # Create CNN encoder
        self.cnn = CNN(input_dim=input_dim, input_channels=input_channels, **encoder_cfg.to_dict())
        cnn_output_dim = self.cnn.output_dim

        # Determine MLP output dimension
        mlp_final_dim = (
            self.distribution.input_dim if self.distribution else output_dim
        )

        # Create MLP head
        self.mlp = MLP(
            input_dim=cnn_output_dim,
            output_dim=mlp_final_dim,
            hidden_dims=cfg.hidden_dims,
            activation=cfg.activation,
        )

        # Initialize distribution weights if provided
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self, x: torch.Tensor, stochastic: bool = True
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, C, H, W).
            stochastic: If True, sample from distribution (if available).

        Returns:
            Output tensor.
        """
        latent = self.get_latent(x)
        mlp_out = self.mlp(latent)

        if self.distribution is None:
            return mlp_out

        if stochastic:
            self.distribution.update(mlp_out)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mlp_out)

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """Extract latent representation from CNN.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            CNN latent tensor.
        """
        return self.cnn(x)

    def update_normalization(self, x: torch.Tensor) -> None:
        """Update observation normalization statistics.

        CNN models do not normalize their inputs, so this is a no-op that keeps the interface identical to
        :class:`MLPModel`.

        Args:
            x: Batch of observations.
        """
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Get mean of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Get std of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Get entropy of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Get raw parameters of output distribution."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log probability of outputs."""
        if self.distribution is None:
            raise RuntimeError("No distribution available")
        return self.distribution.log_prob(outputs)

    def as_jit(self) -> nn.Module:
        """Export model for TorchScript/JIT compilation."""
        return _TorchCNNModel(self)

    def as_onnx(self) -> nn.Module:
        """Export model for ONNX."""
        return _OnnxCNNModel(self)


class _TorchMLPModel(nn.Module):
    """TorchScript-friendly export of MLPModel."""

    def __init__(self, model: MLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)


class _OnnxMLPModel(nn.Module):
    """ONNX-friendly export of MLPModel."""

    def __init__(self, model: MLPModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        out = self.mlp(x)
        return self.deterministic_output(out)


class _TorchCNNModel(nn.Module):
    """TorchScript-friendly export of CNNModel."""

    def __init__(self, model: CNNModel) -> None:
        super().__init__()
        self.cnn = copy.deepcopy(model.cnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference."""
        latent = self.cnn(x)
        out = self.mlp(latent)
        return self.deterministic_output(out)


class _OnnxCNNModel(nn.Module):
    """ONNX-friendly export of CNNModel."""

    def __init__(self, model: CNNModel) -> None:
        super().__init__()
        self.cnn = copy.deepcopy(model.cnn)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = (
                model.distribution.as_deterministic_output_module()
            )
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent = self.cnn(x)
        out = self.mlp(latent)
        return self.deterministic_output(out)
