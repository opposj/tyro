from dataclasses import dataclass, field

from tyro.extras import AtomConfig


@dataclass(frozen=True)
class DispNetSimple(AtomConfig, model="disp", dataset="c10"):
    """Settings for the DispNet model."""

    # The construction pattern for the model.
    pattern: list[int] = field(default_factory=lambda: [1, 4, 5, 3])


@dataclass(frozen=True)
class DispNetConfig[optim](AtomConfig, model="disp"):
    """Settings for the DispNet model."""

    # The sub-configuration for the specified optimizer.
    optim_cfg: optim = None

    # The construction pattern for the model.
    pattern: list[int] = field(default_factory=lambda: [3, 4, 6, 3])


@dataclass(frozen=True)
class DispNetSTConfig[optim](AtomConfig, model="disp", method="st"):
    """Settings for the DispNet model."""

    # The sub-configuration for the specified optimizer.
    optim_cfg: optim = None

    # The construction pattern for the model.
    pattern: list[int] = field(default_factory=lambda: [2, 2, 2, 3])
    # The depth for the `st` method.
    st_depth: int = 2


AtomConfig.register(DispNetSTConfig, model="disp", method="st", dataset="c10")


@dataclass(frozen=True)
class AdamDispNetOptimConfig(AtomConfig, optim="adam", model="disp"):
    """Settings for the Adam optimizer."""

    # The learning rate for the optimizer.
    lr: float = 0.003
    # The weight decay for the optimizer.
    weight_decay: float = 0.5


@dataclass(frozen=True)
class AdamOptimConfig(AtomConfig, optim="adam"):
    """Settings for the Adam optimizer."""

    # The learning rate for the optimizer.
    lr: float = 0.001
    # The weight decay for the optimizer.
    weight_decay: float = 0.0


@dataclass(frozen=True)
class SGDOptimConfig(AtomConfig, optim="sgd"):
    """Settings for the SGD optimizer."""

    # The learning rate for the optimizer.
    lr: float = 0.01
    # The momentum for the optimizer.
    momentum: float = 0.9
    # The weight decay for the optimizer.
    weight_decay: float = 0.0
    # The Nesterov momentum for the optimizer.
    nesterov: bool = False


@dataclass(frozen=True)
class DistNetScheSTConfig[optim,scheduler2](AtomConfig, model="dist"):
    """Settings for the DispNet model."""

    # The sub-configuration for the specified optimizer.
    optim_cfg: optim = None
    # The sub-configuration for the specified scheduler.
    scheduler_cfg: scheduler2 = None

    # The construction pattern for the model.
    pattern: list[int] = field(default_factory=lambda: [2, 2, 2, 3])
    # The depth for the `st` method.
    st_depth: int = 2


@dataclass(frozen=True)
class SGDScheOptimConfig[scheduler](AtomConfig, optim="sgd", method="gt"):
    """Settings for the SGD optimizer."""

    # The sub-configuration for the specified scheduler.
    scheduler_cfg: scheduler = None

    # The learning rate for the optimizer.
    lr: float = 0.01
    # The momentum for the optimizer.
    momentum: float = 0.9
    # The weight decay for the optimizer.
    weight_decay: float = 0.0
    # The Nesterov momentum for the optimizer.
    nesterov: bool = False


@dataclass(frozen=True)
class CosSchedulerConfig(AtomConfig, scheduler="cos"):
    """Settings for the cosine scheduler."""

    # The number of epochs for the scheduler.
    epochs: int = 100
    # The minimum learning rate for the scheduler.
    min_lr: float = 0.0


@dataclass(frozen=True)
class CosSchedulerConfig2(AtomConfig, scheduler2="cos"):
    """Settings for the cosine scheduler."""

    # The number of epochs for the scheduler.
    epochs: int = 100
    # The minimum learning rate for the scheduler.
    min_lr: float = 0.0


@dataclass(frozen=True)
class StepSchedulerConfig(AtomConfig, scheduler="step"):
    """Settings for the step scheduler."""

    # The step size for the scheduler.
    step_size: int = 30
    # The gamma for the scheduler.
    gamma: float = 0.1


@dataclass(frozen=True)
class StepSchedulerConfig2(AtomConfig, scheduler2="step"):
    """Settings for the step scheduler."""

    # The step size for the scheduler.
    step_size: int = 30
    # The gamma for the scheduler.
    gamma: float = 0.1
