from pathlib import Path

import yaml
from omegaconf import DictConfig


def test_storage_config(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig) -> None:
    cfg_path = Path(run_trainings_not_dry) / "config.yaml"

    assert cfg_path.exists()

    with cfg_path.open() as f:
        loaded_cfg = yaml.safe_load(f)
    assert loaded_cfg == cfg_all_not_dry


def test_storage_checkpoint(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig) -> None:
    cktps_path = Path(run_trainings_not_dry) / "checkpoints"

    assert cktps_path.exists()

    checkpoints = list(cktps_path.glob("*"))
    assert len(checkpoints) == 1
