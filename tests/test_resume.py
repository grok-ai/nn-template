from importlib import import_module
from pathlib import Path

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule


def test_load_checkpoint(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig) -> None:
    ckpts_path = Path(run_trainings_not_dry) / "checkpoints"
    checkpoint_path = next(ckpts_path.glob("*"))
    assert checkpoint_path

    reference: str = cfg_all_not_dry.nn.module._target_
    module_ref, class_ref = reference.rsplit(".", maxsplit=1)
    module_class: LightningModule = getattr(import_module(module_ref), class_ref)
    assert module_class is not None

    module = module_class.load_from_checkpoint(checkpoint_path=checkpoint_path)
    assert module is not None
    assert sum(p.numel() for p in module.parameters())


def test_cfg_in_checkpoint(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig) -> None:
    ckpts_path = Path(run_trainings_not_dry) / "checkpoints"
    checkpoint_path = next(ckpts_path.glob("*"))
    assert checkpoint_path

    checkpoint = torch.load(checkpoint_path)
    assert "cfg" in checkpoint
    assert checkpoint["cfg"] == cfg_all_not_dry
