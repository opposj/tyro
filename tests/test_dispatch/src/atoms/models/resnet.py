from dataclasses import dataclass, field

from tyro.extras import AtomConfig


@dataclass(frozen=True)
class ResNetConfig(AtomConfig, model="resnet"):
    """Settings for the ResNet model."""

    # The construction pattern for the model.
    pattern: list[int] = field(default_factory=lambda: [3, 4, 6, 3])


@dataclass(frozen=True)
class STResNetConfig(AtomConfig, model="resnet", method="st"):
    """Settings for the ResNet model. Specified by the ST method."""

    # The construction pattern for the model.
    pattern: list[int] = field(default_factory=lambda: [3, 4, 6, 3])
