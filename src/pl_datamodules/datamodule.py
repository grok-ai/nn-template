from typing import Dict, Optional, Sequence, Tuple, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, ValueNode
from torch.utils.data import DataLoader, Dataset, random_split


class MyDataset(Dataset):
    def __init__(
        self, name: ValueNode, path: ValueNode, train: bool, cfg: DictConfig, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.name = name
        self.train = train

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"MyDataset({self.name=}, {self.path=})"


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_percentage: float,
        cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.val_percentage = val_percentage

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        # download only
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None):

        # split dataset
        if stage is None or stage == "fit":
            mnist_train = hydra.utils.instantiate(self.datasets.train, cfg=self.cfg)
            train_length = int(len(mnist_train) * (1 - self.val_percentage))
            val_length = len(mnist_train) - train_length
            self.train_dataset, self.val_dataset = random_split(
                mnist_train, [train_length, val_length]
            )
        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(x, cfg=self.cfg) for x in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
        )

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                x,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
            )
            for x in self.test_datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
            f"{self.val_percentage=}"
        )
