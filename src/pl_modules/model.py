from typing import Any, Dict, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Optimizer

from src.common.utils import PROJECT_ROOT

class MyModel(pl.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters() # populate self.hparams with args and kwargs automagically!

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.
        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        raise NotImplementedError

    def step(self, batch: Any, batch_idx: int):
        raise NotImplementedError

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"val_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss},
        )
        return loss

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """
        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.
        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]



@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    model: pl.LightningModule = hydra.utils.instantiate(
        cfg.model,
        optim=cfg.optim,
        data=cfg.data,
        logging=cfg.logging,
        _recursive_=False,
    )


if __name__ == "__main__":
    main()
