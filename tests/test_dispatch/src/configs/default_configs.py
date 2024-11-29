from dataclasses import dataclass

from ..atoms.datasets.cifar10 import C10Config, GTC10Config, STMBV3C10Config
from ..atoms.datasets.cifar100 import C100Config
from ..atoms.methods.gt import GTConfig
from ..atoms.methods.st import STConfig
from ..atoms.models.mbv3 import MBV3Config
from ..atoms.models.resnet import ResNetConfig, STResNetConfig
from ..atoms.models.disp_net import DispNetConfig, DispNetSTConfig, AdamOptimConfig, SGDOptimConfig, AdamDispNetOptimConfig
from ..configs.main_config import Config
from tyro.extras import DefaultConfig


@dataclass(frozen=True)
class Default(DefaultConfig):
    default1 = Config(
        x = 2,
    )
    default2 = Config(
        x = 3,
    )


@dataclass(frozen=True)
class StDefault(DefaultConfig, method="st"):
    default1 = Config(
        method_cfg=STConfig(
            main_coef=1.0,
        ),
    )
    default2 = Config(
        method_cfg=STConfig(
            main_coef=2.0,
        ),
    )


@dataclass(frozen=True)
class GtC10Default(DefaultConfig, method="gt", dataset="c10"):
    default1 = Config(
        method_cfg=GTConfig(
            main_coef=1.0,
        ),
        dataset_cfg=C10Config(
            num_classes=10,
        ),
    )
    default2 = Config(
        method_cfg=GTConfig(
            main_coef=2.0,
        ),
        dataset_cfg=C10Config(
            num_classes=20,
        ),
    )
    default3 = Config(
        method_cfg=GTC10Config(
            num_classes=10,
        ),
        dataset_cfg=C10Config(
            num_classes=10,
        ),
    )
    default4 = Config(
        method_cfg=GTC10Config(
            num_classes=20,
        ),
        dataset_cfg=C10Config(
            num_classes=20,
        ),
    )


@dataclass(frozen=True)
class StC100ResNet(DefaultConfig, method="st", dataset="c100", model="resnet"):
    default1 = Config(
        method_cfg=STConfig(
            main_coef=1.0,
        ),
        dataset_cfg=C100Config(
            num_classes=100,
        ),
        model_cfg=ResNetConfig(
            pattern=[1, 1, 1, 1],
        ),
    )
    default2 = Config(
        method_cfg=STConfig(
            main_coef=2.0,
        ),
        dataset_cfg=C100Config(
            num_classes=200,
        ),
        model_cfg=ResNetConfig(
            pattern=[2, 2, 2, 2],
        ),
    )
    default3 = Config(
        method_cfg=STConfig(
            main_coef=2.0,
        ),
        dataset_cfg=C100Config(
            num_classes=200,
        ),
        model_cfg=STResNetConfig(
            pattern=[2, 2, 2, 2],
        ),
    )


@dataclass(frozen=True)
class Mbv3C10Default(DefaultConfig, model="mbv3", dataset="c10"):
    default1 = Config(
        model_cfg=MBV3Config(
            width_mult=1.0,
        ),
        dataset_cfg=C10Config(
            num_classes=10,
        ),
    )
    default2 = Config(
        model_cfg=MBV3Config(
            width_mult=2.0,
        ),
        dataset_cfg=C10Config(
            num_classes=20,
        ),
    )


@dataclass(frozen=True)
class ResNetDefault(DefaultConfig, model="resnet"):
    default1 = Config(
        model_cfg=ResNetConfig(
            pattern=[1, 1, 1, 1],
        ),
    )
    default2 = Config(
        model_cfg=ResNetConfig(
            pattern=[2, 2, 2, 2],
        ),
    )


@dataclass(frozen=True)
class C100Default(DefaultConfig, dataset="c100"):
    default1 = Config(
        dataset_cfg=C100Config(
            num_classes=100,
        ),
    )
    default2 = Config(
        dataset_cfg=C100Config(
            num_classes=200,
        ),
    )


@dataclass(frozen=True)
class StMbv3C10Default(DefaultConfig, method="st", model="mbv3", dataset="c10"):
    default1 = Config(
        method_cfg=STConfig(
            main_coef=1.0,
        ),
        model_cfg=MBV3Config(
            width_mult=1.0,
        ),
        dataset_cfg=STMBV3C10Config(
            num_classes=10,
        ),
    )
    default2 = Config(
        method_cfg=STConfig(
            main_coef=2.0,
        ),
        model_cfg=MBV3Config(
            width_mult=2.0,
        ),
        dataset_cfg=STMBV3C10Config(
            num_classes=20,
        ),
    )


@dataclass(frozen=True)
class DispDefault(DefaultConfig, model="disp"):
    default1 = Config(
        model_cfg=DispNetConfig(
            pattern=[3, 4, 6, 3],
            optim_cfg=AdamOptimConfig(
                lr=0.001,
            ),
        ),
    )
    default2 = Config(
        model_cfg=DispNetConfig(
            pattern=[2, 2, 2, 3],
            optim_cfg=SGDOptimConfig(
                lr=0.01,
                momentum=0.9,
            ),
        ),
    )
    default3 = Config(
        model_cfg=DispNetConfig(
            pattern=[4, 2, 6, 3],
        ),
    )


@dataclass(frozen=True)
class STDispAdamDefault(DefaultConfig, method="st", model="disp", optim="adam"):
    default1 = Config(
        method_cfg=STConfig(
            main_coef=1.0,
        ),
        model_cfg=DispNetSTConfig(
            pattern=[3, 4, 6, 3],
            optim_cfg=AdamOptimConfig(
                lr=0.001,
            ),
        ),
    )
    default2 = Config(
        method_cfg=STConfig(
            main_coef=2.0,
        ),
        model_cfg=DispNetSTConfig(
            pattern=[2, 2, 2, 3],
            optim_cfg=AdamOptimConfig(
                lr=0.001,
            ),
        ),
    )
    default3 = Config(
        method_cfg=STConfig(
            main_coef=1.0,
        ),
        model_cfg=DispNetSTConfig(
            pattern=[3, 4, 6, 3],
            optim_cfg=AdamDispNetOptimConfig(
                lr=0.001,
            ),
        ),
    )
    default4 = Config(
        method_cfg=STConfig(
            main_coef=2.0,
        ),
        model_cfg=DispNetSTConfig(
            pattern=[2, 2, 2, 3],
            optim_cfg=AdamDispNetOptimConfig(
                lr=0.001,
            ),
        ),
    )