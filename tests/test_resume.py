import torch
from omegaconf import DictConfig, OmegaConf
from pytest import TempPathFactory

from tests.conftest import TRAIN_MAX_NSTEPS, get_checkpoint_path, load_checkpoint

from nn_template.run import run


def test_resume(run_trainings_not_dry: str, cfg_all_not_dry: DictConfig, tmp_path_factory: TempPathFactory) -> None:
    old_checkpoint_path = get_checkpoint_path(run_trainings_not_dry)

    new_cfg = OmegaConf.create(cfg_all_not_dry)
    new_storage_dir = tmp_path_factory.mktemp("resumed_training")

    new_cfg.core.storage_dir = str(new_storage_dir)
    new_cfg.train.trainer.max_steps = 2 * TRAIN_MAX_NSTEPS

    new_cfg.train.resume.ckpt_or_run_path = str(old_checkpoint_path)
    new_cfg.train.resume.training = True
    new_cfg.train.resume.logging = False

    new_training_dir = run(new_cfg)

    old_checkpoint = torch.load(old_checkpoint_path)
    new_checkpoint = load_checkpoint(new_training_dir)

    assert old_checkpoint["run_path"] != new_checkpoint["run_path"]
    assert old_checkpoint["global_step"] * 2 == new_checkpoint["global_step"]
    assert new_checkpoint["epoch"] == 2
