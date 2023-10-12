import logging
from functools import cached_property, partial
from pathlib import Path
from typing import List, Mapping, Optional

import hydra
import lightning.pytorch as pl
import omegaconf
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm

from nn_core.common import PROJECT_ROOT
from nn_core.nn_types import Split

pylogger = logging.getLogger(__name__)


class MetaData:
    def __init__(self, class_vocab: Mapping[str, int]):
        """The data information the Lightning Module will be provided with.

        This is a "bridge" between the Lightning DataModule and the Lightning Module.
        There is no constraint on the class name nor in the stored information, as long as it exposes the
        `save` and `load` methods.

        The Lightning Module will receive an instance of MetaData when instantiated,
        both in the train loop or when restored from a checkpoint.

        This decoupling allows the architecture to be parametric (e.g. in the number of classes) and
        DataModule/Trainer independent (useful in prediction scenarios).
        MetaData should contain all the information needed at test time, derived from its train dataset.

        Examples are the class names in a classification task or the vocabulary in NLP tasks.
        MetaData exposes `save` and `load`. Those are two user-defined methods that specify
        how to serialize and de-serialize the information contained in its attributes.
        This is needed for the checkpointing restore to work properly.

        Args:
            class_vocab: association between class names and their indices
        """
        # example
        self.class_vocab: Mapping[str, int] = class_vocab

    def save(self, dst_path: Path) -> None:
        """Serialize the MetaData attributes into the zipped checkpoint in dst_path.

        Args:
            dst_path: the root folder of the metadata inside the zipped checkpoint
        """
        pylogger.debug(f"Saving MetaData to '{dst_path}'")

        # example
        (dst_path / "class_vocab.tsv").write_text(
            "\n".join(f"{key}\t{value}" for key, value in self.class_vocab.items())
        )

    @staticmethod
    def load(src_path: Path) -> "MetaData":
        """Deserialize the MetaData from the information contained inside the zipped checkpoint in src_path.

        Args:
            src_path: the root folder of the metadata inside the zipped checkpoint

        Returns:
            an instance of MetaData containing the information in the checkpoint
        """
        pylogger.debug(f"Loading MetaData from '{src_path}'")

        # example
        lines = (src_path / "class_vocab.tsv").read_text(encoding="utf-8").splitlines()

        class_vocab = {}
        for line in lines:
            key, value = line.strip().split("\t")
            class_vocab[key] = value

        return MetaData(
            class_vocab=class_vocab,
        )

    def __repr__(self) -> str:
        attributes = ",\n    ".join([f"{key}={value}" for key, value in self.__dict__.items()])
        return f"{self.__class__.__name__}(\n    {attributes}\n)"


def collate_fn(samples: List, split: Split, metadata: MetaData):
    """Custom collate function for dataloaders with access to split and metadata.

    Args:
        samples: A list of samples coming from the Dataset to be merged into a batch
        split: The data split (e.g. train/val/test)
        metadata: The MetaData instance coming from the DataModule or the restored checkpoint

    Returns:
        A batch generated from the given samples
    """
    return default_collate(samples)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        accelerator: str,
        # example
        val_images_fixed_idxs: List[int],
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_size = batch_size
        # https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#gpus
        self.pin_memory: bool = accelerator is not None and str(accelerator) == "gpu"

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        # example
        self.val_images_fixed_idxs: List[int] = val_images_fixed_idxs

    @cached_property
    def metadata(self) -> MetaData:
        """Data information to be fed to the Lightning Module as parameter.

        Examples are vocabularies, number of classes...

        Returns:
            metadata: everything the model should know about the data, wrapped in a MetaData object.
        """
        # Since MetaData depends on the training data, we need to ensure the setup method has been called.
        if self.train_dataset is None:
            self.setup(stage="fit")

        return MetaData(class_vocab={i: name for i, name in enumerate(self.train_dataset.features["y"].names)})

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        self.transform = hydra.utils.instantiate(self.dataset.transforms)

        self.hf_datasets = hydra.utils.instantiate(self.dataset)
        self.hf_datasets.set_transform(self.transform)

        # Here you should instantiate your dataset, you may also split the train into train and validation if needed.
        if (stage is None or stage == "fit") and (self.train_dataset is None and self.val_dataset is None):
            self.train_dataset = self.hf_datasets["train"]
            self.val_dataset = self.hf_datasets["val"]

        if stage is None or stage == "test":
            self.test_dataset = self.hf_datasets["test"]

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="train", metadata=self.metadata),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="val", metadata=self.metadata),
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
            pin_memory=self.pin_memory,
            collate_fn=partial(collate_fn, split="test", metadata=self.metadata),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_size=})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig) -> None:
    """Debug main to quickly develop the DataModule.

    Args:
        cfg: the hydra configuration
    """
    m: pl.LightningDataModule = hydra.utils.instantiate(cfg.nn.data, _recursive_=False)
    m.metadata
    m.setup()

    for _ in tqdm(m.train_dataloader()):
        pass


if __name__ == "__main__":
    main()
