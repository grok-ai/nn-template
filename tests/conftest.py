import os

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytest import FixtureRequest, TempdirFactory
from pytorch_lightning import seed_everything

from nn_template.run import run

seed_everything(0)

TRAIN_MAX_NSTEPS = 1


@pytest.fixture(scope="package")
def cfg() -> DictConfig:
    with initialize(config_path="../conf"):
        cfg = compose(config_name="default", return_hydra_config=True)
        HydraConfig().set_config(cfg)
        yield cfg


@pytest.fixture(scope="package")
def cfg_simple_train(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(cfg)

    # Disable logger
    cfg.train.logger.mode = "disabled"

    # Disable multiple workers in test training
    cfg.nn.data.num_workers.train = 0
    cfg.nn.data.num_workers.val = 0
    cfg.nn.data.num_workers.test = 0

    # Minimize the amount of work in test training
    cfg.train.trainer.max_steps = TRAIN_MAX_NSTEPS
    cfg.train.trainer.val_check_interval = TRAIN_MAX_NSTEPS

    return cfg


@pytest.fixture(scope="package")
def cfg_fast_dev_run(cfg_simple_train: DictConfig) -> DictConfig:
    cfg_simple_train = OmegaConf.create(cfg_simple_train)

    # Enable the fast_dev_run flag
    cfg_simple_train.train.trainer.fast_dev_run = True
    return cfg_simple_train


@pytest.fixture(scope="package", params=["cfg_simple_train", "cfg_fast_dev_run"])
def cfg_all_training(request: FixtureRequest):
    return request.getfixturevalue(request.param)


#
# Test training scripts
#
def _test_train(tmpdir_factory: TempdirFactory, cfg_parametrized: DictConfig) -> None:
    cfg_parametrized = OmegaConf.create(cfg_parametrized)

    test_train_tmpdir = tmpdir_factory.mktemp("test_train_tmpdir")

    # Force the wandb dir to be in the temp folder
    os.environ["WANDB_DIR"] = str(test_train_tmpdir)

    cfg_parametrized.core.default_root_dir = str(test_train_tmpdir)

    yield run(cfg=cfg_parametrized)

    test_train_tmpdir.remove()


@pytest.fixture(scope="package")
def run_test_train(tmpdir_factory: TempdirFactory, cfg_all_training: DictConfig):
    yield from _test_train(tmpdir_factory=tmpdir_factory, cfg_parametrized=cfg_all_training)
