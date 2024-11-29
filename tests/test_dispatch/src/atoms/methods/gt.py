from dataclasses import dataclass

from tyro.extras import AtomConfig


@dataclass(frozen=True)
class GTConfig(AtomConfig, method="gt"):
    """Settings for the GT method."""

    # The coefficient for the main loss. e.g. --mtd_cfg.st_main_coef=2.0
    main_coef: float = 0.5
    # The coefficient for the supplementary loss. e.g. --mtd_cfg.st_sup_coef=0.5
    sup_coef: float = 5.0
