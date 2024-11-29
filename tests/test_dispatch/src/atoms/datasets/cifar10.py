from dataclasses import dataclass

from tyro.conf import Fixed

from tyro.extras import AtomConfig


@dataclass(frozen=True)
class C10Config(AtomConfig, dataset="c10"):
    """Settings for the CIFAR-10 dataset."""

    # The number of classes in the dataset.
    num_classes: Fixed[int] = 10


@dataclass(frozen=True)
class GTC10Config(AtomConfig, method="gt", dataset="c10"):
    """Settings for the CIFAR-10 dataset. Specified by the GT method."""

    # The number of classes in the dataset.
    num_classes: Fixed[int] = 10


@dataclass(frozen=True)
class STMBV3C10Config(AtomConfig, dataset="c10", method="st", model="mbv3"):
    """Settings for the CIFAR-10 dataset. Specified by the ST method and the MobileNetV3 model."""

    # The number of classes in the dataset.
    num_classes: Fixed[int] = 10
