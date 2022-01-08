from omegaconf import DictConfig


def test_configuration_parsing(cfg: DictConfig):
    assert cfg is not None
