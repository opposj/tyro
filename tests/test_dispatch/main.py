from src.configs.main_config import Config
from tyro.extras import tyro_dispatch

global_arg = tyro_dispatch(Config, ["src.atoms", "src.configs.default_configs"])


if __name__ == "__main__":
    print(global_arg)

# TODO:
#  让 default变成optional
#  ruff utils；
