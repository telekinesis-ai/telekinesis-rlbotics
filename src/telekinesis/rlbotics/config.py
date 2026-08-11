"""Typed configuration dataclasses for runners, algorithms, models, and distributions.

:class:`~telekinesis.rlbotics.runner.OnPolicyRunner` takes an :class:`OnPolicyRunnerConfig`, which
bundles the algorithm, actor and critic configs. Every config validates itself on construction and
round-trips through plain dictionaries via :meth:`BaseConfig.to_dict` and
:meth:`BaseConfig.from_dict`, so YAML files are validated too. See
``examples/module_examples/config_example.py``.

Attributes:
    ACTIVATIONS: Valid activation function names.
    OPTIMIZERS: Valid optimizer names, matching :data:`~telekinesis.rlbotics.algorithms.OPTIMIZERS`.
    PADDING_MODES: Valid convolution padding modes. ``"none"`` disables padding.
    NORM_TYPES: Valid convolution normalization types.
    GLOBAL_POOL_TYPES: Valid global pooling types.
    SCHEDULES: Valid learning rate schedules.
    STD_TYPES: Valid standard deviation parameterizations.
    TORCH_COMPILE_MODES: Valid :func:`torch.compile` modes, excluding the CUDA-graph ones.
    RESUME_CHOICES: Which checkpoint a resume starts from, the most recent or the best scoring.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar

# Activations resolved by rlbotics.models._resolve_activation(), kept here as a literal set so that
# importing
# configs does not pull in torch.
ACTIVATIONS = frozenset(
    {
        "relu",
        "elu",
        "selu",
        "crelu",
        "lrelu",
        "leaky_relu",
        "tanh",
        "sigmoid",
        "softplus",
        "gelu",
        "swish",
        "mish",
        "identity",
    }
)

OPTIMIZERS = frozenset({"adam", "adamw", "sgd", "rmsprop"})

PADDING_MODES = frozenset({"none", "zeros", "reflect", "replicate", "circular"})

NORM_TYPES = frozenset({"none", "batch", "layer"})

GLOBAL_POOL_TYPES = frozenset({"none", "max", "avg"})

SCHEDULES = frozenset({"adaptive", "fixed"})

STD_TYPES = frozenset({"scalar", "log"})

TORCH_COMPILE_MODES = frozenset({"default", "max-autotune-no-cudagraphs"})

RESUME_CHOICES = frozenset({"last", "best"})


def _to_plain(value: Any) -> Any:
    """Recursively convert configs, and containers of configs, into plain Python objects.

    Args:
        value: Value to convert.

    Returns:
        The value with all nested configs replaced by dictionaries.
    """
    if isinstance(value, BaseConfig):
        return value.to_dict()
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_plain(item) for item in value)
    return value


def _check_positive(name: str, value: Any) -> None:
    """Check that a value is strictly positive.

    Args:
        name: Field name, used in the error message.
        value: Value to check.

    Raises:
        ValueError: If the value is not strictly positive.
    """
    if value <= 0:
        raise ValueError(f"'{name}' must be positive, got {value}.")


def _check_choice(name: str, value: Any, choices: frozenset[str]) -> None:
    """Check that a value is one of the allowed choices.

    Args:
        name: Field name, used in the error message.
        value: Value to check.
        choices: Allowed values.

    Raises:
        ValueError: If the value is not in ``choices``.
    """
    if value not in choices:
        raise ValueError(f"'{name}' must be one of {sorted(choices)}, got {value!r}.")


def _check_unit_interval(name: str, value: float) -> None:
    """Check that a value lies within ``[0, 1]``.

    Args:
        name: Field name, used in the error message.
        value: Value to check.

    Raises:
        ValueError: If the value is outside ``[0, 1]``.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"'{name}' must be in [0, 1], got {value}.")


def _model_config_from_dict(cfg: dict[str, Any]) -> MLPConfig | CNNConfig:
    """Build the model config matching the ``class_name`` in the given dictionary.

    Args:
        cfg: Model configuration dictionary.

    Returns:
        A :class:`CNNConfig` if ``class_name`` resolves to ``CNNModel``, otherwise an
        :class:`MLPConfig`.
    """
    class_name = cfg.get("class_name", "MLPModel")
    if callable(class_name):
        class_name = getattr(class_name, "__name__", "MLPModel")
    simple_name = str(class_name).split(":")[-1].split(".")[-1]
    config_cls = CNNConfig if simple_name == "CNNModel" or "cnn_cfg" in cfg else MLPConfig
    return config_cls.from_dict(cfg)


@dataclass
class BaseConfig:
    """Base class providing dictionary conversion and unknown-key detection for all configs."""

    def to_dict(self, keep_none: bool = False) -> dict[str, Any]:
        """Convert the config into a nested dictionary.

        A key whose value is ``None`` is dropped by default when the field also defaults to
        ``None``, so that optional entries such as ``symmetry_cfg`` are not forwarded to
        constructors that do not accept them. A ``None`` that overrides a non-``None`` default, such
        as ``desired_kl=None``, is always kept, because dropping it would silently restore the
        default on the next :meth:`from_dict`.

        Args:
            keep_none: Whether to keep every key whose value is ``None``, including the optional
                ones.

        Returns:
            The config as a nested dictionary.
        """
        out = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if value is None and f.default is None and not keep_none:
                continue
            out[f.name] = _to_plain(value)
        return out

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]):
        """Build the config from a (possibly nested) dictionary.

        Args:
            cfg: Dictionary of config values. Nested configs may be given as dictionaries or as
                config instances.

        Returns:
            The constructed config.

        Raises:
            ValueError: If the dictionary contains keys that are not fields of this config.
        """
        valid = {f.name for f in fields(cls)}
        unknown = set(cfg) - valid
        if unknown:
            raise ValueError(
                f"Unknown key(s) {sorted(unknown)} for {cls.__name__}. Valid keys are: "
                f"{sorted(valid)}."
            )
        return cls(**dict(cfg.items()))

    def replace(self, **changes: Any):
        """Return a copy of the config with the given fields replaced.

        Args:
            **changes: Field values to override.

        Returns:
            A new, revalidated config.
        """
        return dataclasses.replace(self, **changes)

    def _check_class_name(self) -> None:
        """Validate ``class_name`` against the classes this config can configure.

        Fully qualified names such as ``"rlbotics.algorithms:PPO"`` are accepted, so a custom
        subclass can be named. A class or factory passed directly is accepted as-is, since the
        caller has already resolved it.

        Raises:
            ValueError: If ``class_name`` is neither a callable nor a string naming an accepted
                class.
        """
        expected = getattr(self, "expected_class_names", ())
        if not expected:
            return
        name = self.class_name  # type: ignore[attr-defined]
        if callable(name):
            return
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{type(self).__name__}.class_name must be a non-empty string or a callable, got "
                f"{name!r}."
            )
        if name.split(":")[-1].split(".")[-1] not in expected:
            raise ValueError(
                f"{type(self).__name__}.class_name must name one of {list(expected)}, optionally "
                f"qualified with its"
                f" module, or be the class itself. Got '{name}'."
            )


@dataclass
class GaussianDistributionConfig(BaseConfig):
    """Configuration for :class:`~rlbotics.distributions.GaussianDistribution`.

    Parameterizes a diagonal Gaussian with a state-independent standard deviation.

    Attributes:
        class_name: Distribution class to instantiate, as a name or the class itself.
        init_std: Initial standard deviation. Must be strictly positive and lie within
            ``std_range``.
        std_range: Lower and upper bound the standard deviation is clamped to.
        std_type: Standard deviation parameterization, either ``"scalar"`` or ``"log"``.
        learn_std: Whether the standard deviation is a learnable parameter.
    """

    expected_class_names: ClassVar[tuple[str, ...]] = (
        "GaussianDistribution",
        "SquashedGaussianDistribution",
    )

    class_name: str | Callable = "GaussianDistribution"
    init_std: float = 1.0
    std_range: tuple[float, float] = (1e-6, 1e6)
    std_type: str = "scalar"
    learn_std: bool = True

    def __post_init__(self) -> None:
        """Validate the distribution settings.

        Raises:
            ValueError: If any setting is invalid or if ``init_std`` lies outside ``std_range``.
        """
        self._check_class_name()
        _check_choice("std_type", self.std_type, STD_TYPES)
        _check_positive("init_std", self.init_std)

        if len(self.std_range) != 2:
            raise ValueError(f"'std_range' must be a (min, max) pair, got {self.std_range}.")
        self.std_range = (float(self.std_range[0]), float(self.std_range[1]))
        std_min, std_max = self.std_range

        if std_min <= 0.0:
            raise ValueError(f"'std_range' minimum must be positive, got {std_min}.")
        if std_min >= std_max:
            raise ValueError(
                f"'std_range' minimum must be smaller than its maximum, got {self.std_range}."
            )
        if not std_min <= self.init_std <= std_max:
            raise ValueError(
                f"'init_std' ({self.init_std}) must lie within 'std_range' ({self.std_range})."
            )


@dataclass
class MLPConfig(BaseConfig):
    """Configuration for :class:`~rlbotics.models.MLPModel`.

    Used for both actor and critic models. A critic typically leaves ``distribution_cfg`` as
    ``None`` to obtain a deterministic scalar output.

    Attributes:
        class_name: Model class to instantiate, as a name or the class itself.
        hidden_dims: Hidden layer dimensions. A dimension of ``-1`` is inferred from the input
            dimension.
        activation: Activation function applied after each hidden layer.
        obs_normalization: Whether to normalize observations with running statistics before the MLP.
        distribution_cfg: Output distribution configuration. ``None`` yields a deterministic output.
    """

    expected_class_names: ClassVar[tuple[str, ...]] = ("MLPModel",)

    class_name: str | Callable = "MLPModel"
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256)
    activation: str = "elu"
    obs_normalization: bool = False
    distribution_cfg: GaussianDistributionConfig | None = None

    def __post_init__(self) -> None:
        """Validate the model settings and coerce a nested ``distribution_cfg`` into a config.

        Raises:
            ValueError: If any setting is invalid.
        """
        self._check_class_name()
        _check_choice("activation", self.activation, ACTIVATIONS)

        if len(self.hidden_dims) == 0:
            raise ValueError("'hidden_dims' must contain at least one layer.")
        for dim in self.hidden_dims:
            if dim != -1 and dim <= 0:
                raise ValueError(
                    f"'hidden_dims' entries must be positive or -1 (infer from input), got {dim}."
                )

        if isinstance(self.distribution_cfg, dict):
            self.distribution_cfg = GaussianDistributionConfig.from_dict(self.distribution_cfg)


@dataclass
class CNNEncoderConfig(BaseConfig):
    """Configuration for a single :class:`~rlbotics.models.CNN` encoder.

    Per-layer settings accept either a scalar shared by all layers or a sequence with one entry per
    layer.

    Attributes:
        output_channels: Output channels of each convolutional layer. Its length defines the number
            of layers.
        kernel_size: Convolution kernel size(s).
        stride: Convolution stride(s).
        dilation: Convolution dilation(s).
        padding: Padding mode, one of ``"none"``, ``"zeros"``, ``"reflect"``, ``"replicate"``, or
            ``"circular"``.
        norm: Normalization applied after each convolution, one of ``"none"``, ``"batch"``, or
            ``"layer"``.
        activation: Activation function applied after each layer.
        max_pool: Whether to apply max pooling after each layer.
        global_pool: Global pooling applied to the final feature map, one of ``"none"``, ``"max"``,
            or ``"avg"``.
        flatten: Whether to flatten the output tensor before the MLP head.
    """

    output_channels: tuple[int, ...] | list[int] = (32, 64, 64)
    kernel_size: int | tuple[int, ...] | list[int] = 3
    stride: int | tuple[int, ...] | list[int] = 1
    dilation: int | tuple[int, ...] | list[int] = 1
    padding: str = "none"
    norm: str | tuple[str, ...] | list[str] = "none"
    activation: str = "elu"
    max_pool: bool | tuple[bool, ...] | list[bool] = False
    global_pool: str = "none"
    flatten: bool = True

    def __post_init__(self) -> None:
        """Validate the encoder settings and per-layer sequence lengths.

        Raises:
            ValueError: If any setting is invalid, or if a per-layer sequence does not provide
                exactly one entry per layer.
        """
        _check_choice("padding", self.padding, PADDING_MODES)
        _check_choice("global_pool", self.global_pool, GLOBAL_POOL_TYPES)
        _check_choice("activation", self.activation, ACTIVATIONS)

        num_layers = len(self.output_channels)
        if num_layers == 0:
            raise ValueError("'output_channels' must contain at least one layer.")
        for channels in self.output_channels:
            _check_positive("output_channels", channels)

        # Per-layer parameters: a sequence must provide exactly one entry per layer
        for name in ("kernel_size", "stride", "dilation", "norm", "max_pool"):
            value = getattr(self, name)
            if isinstance(value, (tuple, list)) and len(value) != num_layers:
                raise ValueError(
                    f"'{name}' has {len(value)} entries but there are {num_layers} layers. Provide "
                    f"a scalar or one"
                    f" entry per layer."
                )

        for name in ("kernel_size", "stride", "dilation"):
            value = getattr(self, name)
            for entry in value if isinstance(value, (tuple, list)) else (value,):
                _check_positive(name, entry)

        norms = self.norm if isinstance(self.norm, (tuple, list)) else (self.norm,)
        for entry in norms:
            _check_choice("norm", entry, NORM_TYPES)


@dataclass
class CNNConfig(MLPConfig):
    """Configuration for :class:`~rlbotics.models.CNNModel`.

    Inherits every :class:`MLPConfig` field, which configures the MLP head placed after the CNN
    encoder(s).

    Attributes:
        class_name: Model class to instantiate, as a name or the class itself.
        cnn_cfg: CNN encoder configuration. Either a single encoder config shared by all image
            observations, or a mapping from observation group name to encoder config for per-group
            encoders. ``None`` means no encoder is built.
    """

    expected_class_names: ClassVar[tuple[str, ...]] = ("CNNModel",)

    class_name: str | Callable = "CNNModel"
    cnn_cfg: CNNEncoderConfig | dict[str, CNNEncoderConfig] | None = None

    def __post_init__(self) -> None:
        """Validate the model settings and coerce nested dictionaries into configs.

        Raises:
            ValueError: If any setting is invalid.
        """
        super().__post_init__()

        if isinstance(self.cnn_cfg, dict):
            # Distinguish a single encoder config given as a dict from a mapping of per-group
            # encoder configs
            encoder_keys = {f.name for f in fields(CNNEncoderConfig)}
            if self.cnn_cfg and set(self.cnn_cfg) <= encoder_keys:
                self.cnn_cfg = CNNEncoderConfig.from_dict(self.cnn_cfg)
            else:
                self.cnn_cfg = {
                    group: cfg
                    if isinstance(cfg, CNNEncoderConfig)
                    else CNNEncoderConfig.from_dict(cfg)
                    for group, cfg in self.cnn_cfg.items()
                }


@dataclass
class SymmetryConfig(BaseConfig):
    """Configuration for :class:`~telekinesis.rlbotics.symmetry.Symmetry`, PPO's symmetry extension.

    A robot that is left-right symmetric gives the algorithm a free invariance: a policy that has
    learned to trot leading with one leg has, in principle, learned the mirrored gait too. Saying so
    is worth real sample efficiency, and it is what stops a policy settling into a limp.

    Constructing this config means symmetry is wanted, so both uses of the mirror function are on by
    default. Turning both off leaves the loss computed and reported but detached, which is a cheap way
    to watch how symmetric a policy is without changing what it optimizes.

    Attributes:
        data_augmentation_func: The mirror function, or an import path to it such as
            ``"my_robot.symmetry:mirror"``. Called as ``func(env=env, obs=obs, actions=actions)``,
            returning each argument as the originals stacked with their mirrored copies; either
            argument may be ``None``. It has to come from whoever knows the robot's joint order,
            since only they can say which observation entry mirrors which.
        use_data_augmentation: Whether to append the mirrored samples to every mini-batch, so the
            surrogate and value losses see both.
        use_mirror_loss: Whether to add the mirror loss to the objective, penalizing the policy for
            disagreeing with itself on mirrored observations.
        mirror_loss_coeff: Weight of the mirror loss. ``0`` leaves the term without effect.
    """

    data_augmentation_func: Callable | str | None = None
    use_data_augmentation: bool = True
    use_mirror_loss: bool = True
    mirror_loss_coeff: float = 1.0

    def __post_init__(self) -> None:
        """Validate the symmetry settings.

        Raises:
            ValueError: If no mirror function was given, or the coefficient is negative.
        """
        if self.data_augmentation_func is None:
            raise ValueError(
                "'data_augmentation_func' is required: symmetry needs a mirror function to know "
                "which observation entry mirrors which. Pass the function, or an import path to it "
                "such as 'my_robot.symmetry:mirror'."
            )
        if self.mirror_loss_coeff < 0.0:
            raise ValueError(
                f"'mirror_loss_coeff' must be non-negative, got {self.mirror_loss_coeff}."
            )


@dataclass
class PPOConfig(BaseConfig):
    """Configuration for :class:`~rlbotics.algorithms.PPO`.

    Attributes:
        class_name: Algorithm class to instantiate, as a name or the class itself.
        optimizer: Optimizer for the actor and critic parameters.
        learning_rate: Initial learning rate. Adapted at runtime when ``schedule`` is
            ``"adaptive"``.
        num_learning_epochs: Number of optimization epochs over the collected rollout.
        num_mini_batches: Number of mini-batches the rollout is split into per epoch.
        schedule: Learning rate schedule, either ``"adaptive"`` (KL-based) or ``"fixed"``.
        value_loss_coef: Weight of the value function loss.
        clip_param: PPO surrogate clipping parameter.
        use_clipped_value_loss: Whether to clip the value function loss.
        desired_kl: Target KL divergence for the adaptive schedule. ``None`` disables learning rate
            adaptation.
        entropy_coef: Weight of the entropy bonus.
        gamma: Discount factor.
        lam: Generalized advantage estimation (GAE) lambda.
        max_grad_norm: Maximum gradient norm used for gradient clipping.
        normalize_advantage_per_mini_batch: Whether to normalize advantages per mini-batch instead
            of over the whole rollout.
        share_cnn_encoders: Whether the critic reuses the actor's CNN encoders.
        rnd_cfg: Random Network Distillation (RND) extension settings. ``None`` disables the
            extension.
        symmetry_cfg: A :class:`SymmetryConfig`, or ``None`` to disable the extension. A plain
            dictionary is accepted and converted.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347
        (2017).
    """

    expected_class_names: ClassVar[tuple[str, ...]] = ("PPO",)

    class_name: str | Callable = "PPO"
    optimizer: str = "adam"
    learning_rate: float = 1e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    schedule: str = "adaptive"
    value_loss_coef: float = 1.0
    clip_param: float = 0.2
    use_clipped_value_loss: bool = True
    desired_kl: float | None = 0.01
    entropy_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    share_cnn_encoders: bool = False
    rnd_cfg: dict[str, Any] | None = None
    symmetry_cfg: SymmetryConfig | dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate the algorithm settings.

        Raises:
            ValueError: If any setting is invalid, or if ``desired_kl`` is ``None`` under the
                adaptive schedule.
        """
        self._check_class_name()
        _check_choice("optimizer", self.optimizer.lower(), OPTIMIZERS)
        _check_choice("schedule", self.schedule, SCHEDULES)

        _check_positive("learning_rate", self.learning_rate)
        _check_positive("num_learning_epochs", self.num_learning_epochs)
        _check_positive("num_mini_batches", self.num_mini_batches)
        _check_positive("clip_param", self.clip_param)
        _check_positive("max_grad_norm", self.max_grad_norm)

        _check_unit_interval("gamma", self.gamma)
        _check_unit_interval("lam", self.lam)

        if self.value_loss_coef < 0.0:
            raise ValueError(f"'value_loss_coef' must be non-negative, got {self.value_loss_coef}.")
        if self.entropy_coef < 0.0:
            raise ValueError(f"'entropy_coef' must be non-negative, got {self.entropy_coef}.")

        if self.desired_kl is not None:
            _check_positive("desired_kl", self.desired_kl)
        elif self.schedule == "adaptive":
            raise ValueError("'desired_kl' must be set when 'schedule' is 'adaptive'.")

        if isinstance(self.symmetry_cfg, dict):
            self.symmetry_cfg = SymmetryConfig.from_dict(self.symmetry_cfg)


@dataclass
class LoggerConfig(BaseConfig):
    """Configuration for :class:`~telekinesis.rlbotics.logger.Logger` and its checkpoints.

    A run is written to ``<log_dir>/<experiment>/<timestamp>/``, and everything belonging to it sits
    flat in that directory: the event file, the configuration dump, the checkpoints and any
    animations recorded next to them. Each run gets its own timestamp, so runs never overwrite one
    another, while ``resume`` looks across the whole experiment to find where training left off.

    Attributes:
        enabled: Whether to log at all. When False the runner trains without writing anything.
        log_name: Logger class to instantiate, as a name or the class itself.
        log_dir: Root directory for all experiments. ``None`` writes nothing, so metrics are
            tracked in memory and printed but no run directory is created.
        experiment: Name of this experiment, the directory grouping its runs.
        log_interval: Iterations between console reports. Metrics still go to the event file every
            iteration, so the curves keep their resolution; this only throttles the printing.
        log_metrics: Whether to write scalars to the event file.
        log_video: Whether to record video or animations of the policy. The runner does not render
            anything itself, it is the training script that reads this and decides what to capture.
        log_config: Whether to dump the resolved configuration as ``config.json``.
        save_interval: Iterations between checkpoints.
        keep_last_n: How many checkpoints to keep in the run directory. Older ones are deleted along
            with any sibling files sharing their name, such as an animation recorded next to them.
            ``0`` keeps all of them. ``model_best.pt`` is never rotated away.
        save_best: Whether to keep ``model_best.pt``, rewritten whenever the mean episode reward
            beats every earlier iteration's. Unlike the periodic checkpoints it does not depend on
            ``save_interval``, so the best policy survives even between two scheduled saves.
        resume: Which checkpoint to continue from, or None to train from scratch. ``"last"`` takes
            the most recently written checkpoint, ``"best"`` the highest mean reward, and anything
            else is read as a checkpoint to load: either a path, or a bare file name such as
            ``"model_100.pt"`` to look up among the experiment's runs. The search always covers every
            run of the experiment, so training continues from wherever it left off rather than from
            this run's empty directory.
    """

    expected_class_names: ClassVar[tuple[str, ...]] = ("Logger",)

    enabled: bool = True
    log_name: str | Callable = "Logger"
    log_dir: str | None = "logs"
    experiment: str = "experiment"

    log_interval: int = 100
    log_metrics: bool = True
    log_video: bool = False
    log_config: bool = True

    save_interval: int = 1000
    keep_last_n: int = 5
    save_best: bool = True
    resume: str | None = None

    def __post_init__(self) -> None:
        """Validate the logging and checkpoint settings.

        Raises:
            ValueError: If an interval is not positive, ``keep_last_n`` is negative, or ``log_name``
                names no known logger.
        """
        # The class-name check reads `class_name`, so alias it for the shared helper
        self.class_name = self.log_name
        self._check_class_name()
        del self.class_name

        _check_positive("log_interval", self.log_interval)
        _check_positive("save_interval", self.save_interval)
        if self.keep_last_n < 0:
            raise ValueError(f"'keep_last_n' must be zero or more, got {self.keep_last_n}.")

        # Anything that is not one of the keywords names a checkpoint, and whether it exists can
        # only be answered when the run directory is known, so CheckpointManager reports that

    @property
    def experiment_dir(self) -> str | None:
        """Directory holding every run of this experiment, or None when nothing is written."""
        if self.log_dir is None:
            return None
        return str(Path(self.log_dir) / self.experiment)


@dataclass
class OnPolicyRunnerConfig(BaseConfig):
    """Configuration for :class:`~rlbotics.runner.OnPolicyRunner`.

    Bundles the observation mapping, rollout settings, and the algorithm, actor, and critic configs.

    Attributes:
        obs_groups: Mapping from observation set (``"actor"``, ``"critic"``) to the environment
            observation groups it consumes.
        num_learning_iterations: Iterations to train for, which
            :meth:`~telekinesis.rlbotics.runner.OnPolicyRunner.learn` uses when its caller does not
            say otherwise.
        num_steps_per_env: Environment steps collected per environment per learning iteration.
        verbose: Whether the runner prints training progress. The cadence comes from
            ``logger.log_interval``; this switches the printing on and off.
        check_for_nan: Whether to check environment outputs for NaN values. Each check forces a
            GPU-to-CPU sync (a Python ``if`` on a CUDA tensor), so leaving this on every step can
            noticeably cap GPU utilization on GPU-resident envs (e.g. mjlab, Isaac Lab). Enable it
            while debugging a new environment/reward function, disable for normal training runs.
        torch_compile_mode: :func:`torch.compile` mode, one of ``None`` (disabled), ``"default"``,
            or ``"max-autotune-no-cudagraphs"``.
        logger: Logging, checkpointing and resume settings.
        algorithm: Algorithm settings.
        actor: Actor model settings. Requires a ``distribution_cfg`` to sample stochastic actions.
        critic: Critic model settings. Outputs a deterministic scalar value estimate.
    """

    obs_groups: dict[str, list[str]]
    num_learning_iterations: int = 1000
    num_steps_per_env: int = 24
    verbose: bool = True
    check_for_nan: bool = False
    torch_compile_mode: str | None = None
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    algorithm: PPOConfig = field(default_factory=PPOConfig)
    actor: MLPConfig | CNNConfig = field(
        default_factory=lambda: MLPConfig(distribution_cfg=GaussianDistributionConfig())
    )
    critic: MLPConfig | CNNConfig = field(default_factory=MLPConfig)

    def __post_init__(self) -> None:
        """Validate the runner settings and coerce nested dictionaries into configs.

        Raises:
            ValueError: If any setting is invalid, if ``obs_groups`` is empty or maps a set to an
                empty list, or if the actor has no ``distribution_cfg``.
        """
        _check_positive("num_learning_iterations", self.num_learning_iterations)
        _check_positive("num_steps_per_env", self.num_steps_per_env)

        if self.torch_compile_mode is not None:
            _check_choice("torch_compile_mode", self.torch_compile_mode, TORCH_COMPILE_MODES)

        if not self.obs_groups:
            raise ValueError(
                "'obs_groups' must not be empty. Provide e.g. {'actor': ['policy'], 'critic': "
                "['policy']}."
            )
        for set_name, groups in self.obs_groups.items():
            if not isinstance(groups, (list, tuple)) or len(groups) == 0:
                raise ValueError(
                    f"Observation set '{set_name}' must map to a non-empty list of observation "
                    f"groups."
                )

        if isinstance(self.logger, dict):
            self.logger = LoggerConfig.from_dict(self.logger)
        if isinstance(self.algorithm, dict):
            self.algorithm = PPOConfig.from_dict(self.algorithm)
        if isinstance(self.actor, dict):
            self.actor = _model_config_from_dict(self.actor)
        if isinstance(self.critic, dict):
            self.critic = _model_config_from_dict(self.critic)

        if self.actor.distribution_cfg is None:
            raise ValueError(
                "The actor requires a 'distribution_cfg' (e.g. GaussianDistributionConfig()) to "
                "sample actions."
            )
