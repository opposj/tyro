from dataclasses import dataclass

from tyro.conf import Fixed

from tyro.extras import AtomConfig


@dataclass(frozen=True)
class C100Config(AtomConfig, dataset="c100"):
    """Settings for the CIFAR-100 dataset."""

    # The number of classes in the dataset.
    num_classes: Fixed[int] = 100
