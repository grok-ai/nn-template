import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig
from pytorch_lightning import seed_everything

from nn_template.common.utils import PROJECT_ROOT

seed_everything(0)


@pytest.fixture
def cfg() -> DictConfig:
    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "conf")):
        return compose(config_name="default")
