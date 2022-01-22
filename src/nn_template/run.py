import logging
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback

from nn_core.callbacks import NNTemplateCore
from nn_core.common import PROJECT_ROOT
from nn_core.common.utils import enforce_tags, seed_index_everything
from nn_core.model_logging import NNLogger
from nn_core.resume import parse_restore

# Force the execution of __init__.py if this file is executed directly.
import nn_template  # noqa

pylogger = logging.getLogger(__name__)


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = [NNTemplateCore()]

    for callback in cfg:
        pylogger.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> str:
    """Generic train loop.

    :param cfg: run configuration, defined by Hydra in /conf
    """
    seed_index_everything(cfg.train)

    fast_dev_run: bool = cfg.train.trainer.fast_dev_run
    if fast_dev_run:
        pylogger.info(f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!")
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

    cfg.core.tags = enforce_tags(cfg.core.get("tags", None))
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

    cfg.train.logging.logger.name = f"{datamodule.name}-{model.name}"
    logger: NNLogger = NNLogger(logging_cfg=cfg.train.logging, cfg=cfg, resume_id=resume_run_version)

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
