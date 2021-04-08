from typing import Dict, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset

from src.common.utils import PROJECT_ROOT


class MyDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, **kwargs):
        super().__init__()
        self.path = path
        self.name = name

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
