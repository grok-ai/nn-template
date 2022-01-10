# Required workaround because PyTorch Lightning configures the logging on import,
# thus the logging configuration defined in the __init__.py must be called before
# the lightning import otherwise it has no effect.
# See https://github.com/PyTorchLightning/pytorch-lightning/issues/1503
#
# Force the execution of __init__.py if this file is executed directly.
import logging
from operator import xor
from typing import List, Optional, Tuple

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, seed_everything
from pytorch_lightning.loggers.base import DummyLogger

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger
from nn_core.resume import resolve_ckpt, resolve_run_path, resolve_run_version

import nn_template  # isort:skip # noqa


pylogger = logging.getLogger(__name__)


RESUME_MODES = {
    "continue": {
        "restore_model": True,
        "restore_run": True,
    },
    "hotstart": {
        "restore_model": True,
        "restore_run": False,
    },
    None: {
        "restore_model": False,
        "restore_run": False,
    },
}


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def parse_restore(restore_cfg: DictConfig) -> Tuple[Optional[str], Optional[str]]:
    ckpt_or_run_path = restore_cfg.ckpt_or_run_path
    resume_mode = restore_cfg.mode

    resume_ckpt_path = None
    resume_run_version = None

    if xor(bool(ckpt_or_run_path), bool(resume_mode)):
        pylogger.warning(f"Inconsistent resume modality {resume_mode} and checkpoint path '{ckpt_or_run_path}'")

    if resume_mode not in RESUME_MODES:
        message = f"Unsupported resume mode {resume_mode}. Available resume modes are: {RESUME_MODES}"
        pylogger.error(message)
        raise ValueError(message)

    flags = RESUME_MODES[resume_mode]
    restore_model = flags["restore_model"]
    restore_run = flags["restore_run"]

    if ckpt_or_run_path is not None:
        if restore_model:
            resume_ckpt_path = resolve_ckpt(ckpt_or_run_path)
            pylogger.info(f"Resume training from: '{resume_ckpt_path}'")

        if restore_run:
            run_path = resolve_run_path(ckpt_or_run_path)
            resume_run_version = resolve_run_version(run_path=run_path)
            pylogger.info(f"Resume logging to: '{run_path}'")

    return resume_ckpt_path, resume_run_version


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    resume_ckpt_path, resume_run_version = parse_restore(cfg.train.restore)

    # Instantiate datamodule
    pylogger.info(f"Instantiating <{cfg.nn.data['_target_']}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    # Instantiate model
    pylogger.info(f"Instantiating <{cfg.nn.module['_target_']}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.nn.module, _recursive_=False)

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg.train.callbacks)

    storage_dir: str = cfg.core.storage_dir

    # The logger attribute will be filled by the NNLoggerConfiguration callback.
    logger: NNLogger = NNLogger(logger=DummyLogger(), storage_dir=storage_dir, cfg=cfg, resume_id=resume_run_version)

    pylogger.info("Instantiating the <Trainer>")
    trainer = pl.Trainer(
        default_root_dir=storage_dir,
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )

    pylogger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=resume_ckpt_path)

    if fast_dev_run:
        pylogger.info("Skipping testing in 'fast_dev_run' mode!")
    else:
        pylogger.info("Starting testing!")
        trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()

    return logger.run_dir


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
