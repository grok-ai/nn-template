import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Union

import hydra
import omegaconf
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

from nn_core.common import PROJECT_ROOT

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_vocab: Mapping[str, int]):
        self.class_vocab: Mapping[str, int] = class_vocab

    def save(self, dst_path: Path):
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        (dst_path / "class_vocab.tsv").write_text(
            "\n".join(f"{key}\t{value}" for key, value in self.class_vocab.items())
        )

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        lines = (src_path / "class_vocab.tsv").read_text(encoding="utf-8").splitlines()

        class_vocab = {}
        for line in lines:
            key, value = line.strip().split("\t")
            class_vocab[key] = value

        return MetaData(
            class_vocab=class_vocab,
        )


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        gpus: Optional[Union[List[int], str, int]],
        # example
        val_percentage: float,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = gpus is not None and str(gpus) != "0"
        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        # example
        self.val_percentage: float = val_percentage

    @property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        return MetaData(class_vocab=self.train_dataset.dataset.class_vocab)

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_datasets is None):
            # example
            mnist_train = hydra.utils.instantiate(
                self.datasets.train,
                split="train",
                transform=transform,
                path=PROJECT_ROOT / "data",
            )
            train_length = int(len(mnist_train) * (1 - self.val_percentage))
            val_length = len(mnist_train) - train_length
            self.train_dataset, val_dataset = random_split(mnist_train, [train_length, val_length])

            self.val_datasets = [val_dataset]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(
                    dataset_cfg,
                    split="test",
                    path=PROJECT_ROOT / "data",
                    transform=transform,
                )
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                pin_memory=self.pin_memory,
            )
            for dataset in self.val_datasets
        ]

    def test_dataloader(self) -> Sequence[DataLoader]:
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                pin_memory=self.pin_memory,
            )
            for dataset in self.test_datasets
        ]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.datasets=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    _: pl.LightningDataModule = hydra.utils.instantiate(cfg.data.datamodule, _recursive_=False)


if __name__ == "__main__":
    main()
