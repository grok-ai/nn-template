import hydra
from omegaconf import DictConfig

from {{ cookiecutter.package_name }}.run import build_callbacks


def test_configuration_parsing(cfg: DictConfig) -> None:
    assert cfg is not None


def test_callbacks_instantiation(cfg: DictConfig) -> None:
    build_callbacks(cfg.train.callbacks)


def test_model_instantiation(cfg: DictConfig) -> None:
    datamodule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    hydra.utils.instantiate(cfg.nn.module, metadata=datamodule.metadata, _recursive_=False)


def test_cfg_parametrization(cfg_all: DictConfig):
    assert cfg_all
