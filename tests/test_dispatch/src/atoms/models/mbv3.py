from dataclasses import dataclass

from tyro.extras import AtomConfig


@dataclass(frozen=True)
class MBV3Config(AtomConfig, model="mbv3"):
    """Settings for the MobileNetV3 model."""

    # The width multiplier for the model.
    width_mult: float = 1.0
