import hydra
from omegaconf import DictConfig

from nn_template.run import build_callbacks


def test_configuration_parsing(cfg: DictConfig) -> None:
    assert cfg is not None


def test_callbacks_instantiation(cfg: DictConfig) -> None:
    build_callbacks(cfg.train.callbacks)


def test_datamodule_instantiation(cfg: DictConfig) -> None:
    hydra.utils.instantiate(cfg.nn.data, _recursive_=False)


def test_pl_module_instantiation(cfg: DictConfig) -> None:
    hydra.utils.instantiate(cfg.nn.module, _recursive_=False)


def test_logger_instantiation(cfg: DictConfig) -> None:
    hydra.utils.instantiate(cfg.train.logger, mode="disabled", _recursive_=False)


def test_cfg_parametrization(cfg_all_training: DictConfig):
    assert cfg_all_training
