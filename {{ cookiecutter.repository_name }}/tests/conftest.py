import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Union

import pytest
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pytest import FixtureRequest, TempPathFactory
from lightning.pytorch import seed_everything

from nn_core.serialization import NNCheckpointIO

from {{ cookiecutter.package_name }}.run import run

logging.basicConfig(force=True, level=logging.DEBUG)

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

    # Add test tag
    cfg.core.tags = ["testing"]

    # Disable gpus
    cfg.train.trainer.accelerator = "cpu"

    # Disable logger
    cfg.train.logging.logger.mode = "disabled"

    # Disable files upload because wandb in offline modes uses always /tmp
    # as run.dir, which causes conflicts between multiple trainings
    cfg.train.logging.upload.run_files = False

    # Disable multiple workers in test training
    cfg.nn.data.num_workers.train = 0
    cfg.nn.data.num_workers.val = 0
    cfg.nn.data.num_workers.test = 0

    # Minimize the amount of work in test training
    cfg.train.trainer.max_steps = TRAIN_MAX_NSTEPS
    cfg.train.trainer.val_check_interval = TRAIN_MAX_NSTEPS

    # Ensure the resuming is disabled
    with open_dict(config=cfg):
        cfg.train.restore = {}
        cfg.train.restore.ckpt_or_run_path = None
        cfg.train.restore.mode = None

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


#
# Utility functions
#
def get_checkpoint_path(storagedir: Union[str, Path]) -> Path:
    ckpts_path = Path(storagedir) / "checkpoints"
    checkpoint_path = next(ckpts_path.glob("*"))
    assert checkpoint_path
    return checkpoint_path


def load_checkpoint(storagedir: Union[str, Path]) -> Dict:
    checkpoint = NNCheckpointIO.load(path=get_checkpoint_path(storagedir))
    assert checkpoint
    return checkpoint
