from pathlib import Path
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback, seed_everything

from nn_template.common.utils import PROJECT_ROOT, log_hyperparameters


def build_callbacks(cfg: DictConfig) -> List[Callback]:
    callbacks: List[Callback] = []

    for callback in cfg:
        hydra.utils.log.info(f"Adding callback <{callback['_target_'].split('.')[-1]}>")
        callbacks.append(hydra.utils.instantiate(callback, _recursive_=False))

    return callbacks


def run(cfg: DictConfig) -> None:
    """Generic train loop.

    :param cfg: run configuration, defined by Hydra in /conf
    """
    if cfg.train.deterministic:
        seed_everything(cfg.train.random_seed)

    if cfg.train.trainer.fast_dev_run:
        hydra.utils.log.info(
            f"Debug mode <{cfg.train.trainer.fast_dev_run=}>. Forcing debugger friendly configuration!"
        )
        # Debuggers don't like GPUs nor multiprocessing
        cfg.train.trainer.gpus = 0
        cfg.nn.data.num_workers.train = 0
        cfg.nn.data.num_workers.val = 0
        cfg.nn.data.num_workers.test = 0

        # Switch wandb mode to offline to prevent online logging
        cfg.train.logger.mode = "offline"

    # Hydra run directory
    hydra_dir = Path(HydraConfig.get().run.dir)

    # Instantiate datamodule
    hydra.utils.log.info(f"Instantiating <{cfg.nn.data._target_}>")
    datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)

    # Instantiate model
    hydra.utils.log.info(f"Instantiating <{cfg.nn.module._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.nn.module,
        _recursive_=False,
    )

    # Instantiate the callbacks
    callbacks: List[Callback] = build_callbacks(cfg=cfg.train.callbacks)

    # Logger instantiation/configuration
    logger = None
    if "logger" in cfg.train:
        logger_cfg = cfg.train.logger
        hydra.utils.log.info(f"Instantiating <{logger_cfg['_target_'].split('.')[-1]}>")
        logger = hydra.utils.instantiate(logger_cfg)

        # TODO: incompatible with other loggers! :]
        if "wandb_watch" in cfg.train:
            hydra.utils.log.info(f"W&B is now watching <{cfg.train.wandb_watch.log}>!")
            logger.watch(
                model,
                log=cfg.train.wandb_watch.log,
                log_freq=cfg.train.wandb_watch.log_freq,
            )

    # Store the YaML config separately into the wandb dir
    yaml_conf: str = OmegaConf.to_yaml(cfg=cfg)
    (Path(logger.experiment.dir) / "hparams.yaml").write_text(yaml_conf)

    hydra.utils.log.info("Instantiating the Trainer")

    # The Lightning core, the Trainer
    trainer = pl.Trainer(
        default_root_dir=str(hydra_dir),
        logger=logger,
        callbacks=callbacks,
        **cfg.train.trainer,
    )
    log_hyperparameters(trainer=trainer, model=model, cfg=cfg)

    hydra.utils.log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    hydra.utils.log.info("Starting testing!")
    trainer.test(datamodule=datamodule)

    # Logger closing to release resources/avoid multi-run conflicts
    if logger is not None:
        logger.experiment.finish()


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    run(cfg)


if __name__ == "__main__":
    main()
