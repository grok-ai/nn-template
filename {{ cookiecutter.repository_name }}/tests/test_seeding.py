import pytest
from omegaconf import DictConfig, OmegaConf

from nn_core.common.utils import seed_index_everything


@pytest.mark.parametrize(
    "seed_index, expected_seed",
    [
        (0, 1608637542),
        (30, 787716372),
    ],
)
def test_seed_index_determinism(cfg_all: DictConfig, seed_index: int, expected_seed: int):
    cfg_all = OmegaConf.create(cfg_all)

    cfg_all.train.seed_index = seed_index
    current_seed = seed_index_everything(train_cfg=cfg_all.train, sampling_seed=42)
    assert current_seed == expected_seed
