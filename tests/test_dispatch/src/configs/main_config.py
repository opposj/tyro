from dataclasses import dataclass


@dataclass(frozen=True)
class Config[method, model, dataset]:
    """
    The only global configuration used everywhere in the project.
    """

    method_cfg: method = None
    """The sub-configuration for the specified method."""

    model_cfg: model = None
    """The sub-configuration for the specified model."""

    dataset_cfg: dataset = None
    """The sub-configuration for the specified dataset."""

    x: int = 1

#
# def easy_init(*args, **kwargs):
#     try:
#         return Config.__init__(*args, **kwargs)
#     except:
#         return None
#
#
# Config.__init__ = easy_init