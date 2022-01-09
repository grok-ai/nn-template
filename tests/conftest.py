import os
import shutil

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytest import FixtureRequest, TempPathFactory
from pytorch_lightning import seed_everything

from nn_template.run import run

seed_everything(42)

TRAIN_MAX_NSTEPS = 1


#
# Base configurations
#
@pytest.fixture(scope="package")
def cfg(tmp_path_factory: TempPathFactory) -> DictConfig:
    test_cfg_tmpdir = tmp_path_factory.mktemp("test_train_tmpdir")

    with initialize(config_path="../conf"):
        cfg = compose(config_name="default", return_hydra_config=True)
        HydraConfig().set_config(cfg)

        # Force the wandb dir to be in the temp folder
        os.environ["WANDB_DIR"] = str(test_cfg_tmpdir)

        # Force the storage dir to be in the temp folder
        cfg.core.storage_dir = str(test_cfg_tmpdir)

        yield cfg

    shutil.rmtree(test_cfg_tmpdir)


#
# Training configurations
#
@pytest.fixture(scope="package")
def cfg_simple_train(cfg: DictConfig) -> DictConfig:
    cfg = OmegaConf.create(cfg)

    # Disable logger
    logger_reference = "nn_core.callbacks.NNLoggerConfiguration"
    callbacks_targets = [x["_target_"] for x in cfg.train.callbacks]
    if logger_reference in callbacks_targets:
        cfg.train.callbacks[callbacks_targets.index(logger_reference)].logger.mode = "disabled"

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


#
# Training configurations aggregations
#
@pytest.fixture(
    scope="package",
    params=[
        "cfg_simple_train",
    ],
)
def cfg_all_not_dry(request: FixtureRequest):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    scope="package",
    params=[
        "cfg_simple_train",
        "cfg_fast_dev_run",
    ],
)
def cfg_all(request: FixtureRequest):
    return request.getfixturevalue(request.param)


#
# Training fixtures
#
@pytest.fixture(
    scope="package",
)
def run_trainings_not_dry(cfg_all_not_dry: DictConfig) -> str:
    yield run(cfg=cfg_all_not_dry)


@pytest.fixture(
    scope="package",
)
def run_trainings(cfg_all: DictConfig) -> str:
    yield run(cfg=cfg_all)
