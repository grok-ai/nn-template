from typing import Union, Dict, Tuple

import torch
from omegaconf import ValueNode, DictConfig
from torch.utils.data import Dataset


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
