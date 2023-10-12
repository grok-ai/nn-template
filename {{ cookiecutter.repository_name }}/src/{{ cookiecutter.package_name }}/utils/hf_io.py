from collections import namedtuple
from pathlib import Path
from typing import Any, Callable, Dict, Sequence

import torch
from anypy.data.metadata_dataset_dict import MetadataDatasetDict
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from omegaconf import DictConfig

from nn_core.common import PROJECT_ROOT

DatasetParams = namedtuple("DatasetParams", ["name", "fine_grained", "train_split", "test_split", "hf_key"])


class HFTransform:
    def __init__(
        self,
        key: str,
        transform: Callable[[torch.Tensor], torch.Tensor],
    ):
        """Apply a row-wise transform to a dataset column.

        Args:
            key (str): The key of the column to transform.
            transform (Callable[[torch.Tensor], torch.Tensor]): The transform to apply.
        """
        self.transform = transform
        self.key = key

    def __call__(self, samples: Dict[str, Sequence[Any]]) -> Dict[str, Sequence[Any]]:
        """Apply the transform to the samples.

        Args:
            samples (Dict[str, Sequence[Any]]): The samples to transform.

        Returns:
            Dict[str, Sequence[Any]]: The transformed samples.
        """
        samples[self.key] = [self.transform(data) for data in samples[self.key]]
        return samples

    def __repr__(self) -> str:
        return repr(self.transform)


def preprocess_dataset(
    dataset: Dataset,
    cfg: Dict,
) -> Dataset:
    """Preprocess a dataset.

    This function applies the following preprocessing steps:
        - Rename the label column to the standard key.
        - Rename the data column to the standard key.

    Do not apply transforms here, as the preprocessed dataset will be saved to disk once
    and then resued; thus updates on the transforms will not be reflected in the dataset.

    Args:
        dataset (Dataset): The dataset to preprocess.
        cfg (Dict): The configuration.

    Returns:
        Dataset: The preprocessed dataset.
    """
    dataset = dataset.rename_column(cfg["label_key"], cfg["standard_y_key"])
    dataset = dataset.rename_column(cfg["data_key"], cfg["standard_x_key"])
    return dataset


def save_dataset_to_disk(dataset: MetadataDatasetDict, output_path: Path) -> None:
    """Save a dataset to disk.

    Args:
        dataset (MetadataDatasetDict): The dataset to save.
        output_path (Path): The path to save the dataset to.
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    dataset.save_to_disk(output_path)


def load_hf_dataset(**cfg: DictConfig) -> MetadataDatasetDict:
    """Load a dataset from the HuggingFace datasets library.

    The returned dataset is a MetadataDatasetDict, which is a wrapper around a DatasetDict.
    It will contain the following splits:
        - train
        - val
        - test
    If `val_split` is not specified in the config, it will be created from the train split
    according to the `val_percentage` specified in the config.

    The returned dataset will be preprocessed and saved to disk,
    if it does not exist yet, and loaded from disk otherwise.

    Args:
        cfg: The configuration.

    Returns:
        Dataset: The loaded dataset.
    """
    dataset_params: DatasetParams = DatasetParams(
        cfg["ref"],
        None,
        cfg["train_split"],
        cfg["test_split"],
        (cfg["ref"],),
    )
    DATASET_KEY = "_".join(
        map(
            str,
            [v for k, v in dataset_params._asdict().items() if k != "hf_key" and v is not None],
        )
    )
    DATASET_DIR: Path = PROJECT_ROOT / "data" / "datasets" / DATASET_KEY

    if not DATASET_DIR.exists():
        train_dataset = load_dataset(
            dataset_params.name,
            split=dataset_params.train_split,
            token=True,
        )
        if "val_percentage" in cfg:
            train_val_dataset = train_dataset.train_test_split(test_size=cfg["val_percentage"], shuffle=True)
            train_dataset = train_val_dataset["train"]
            val_dataset = train_val_dataset["test"]
        elif "val_split" in cfg:
            val_dataset = load_dataset(
                dataset_params.name,
                split=cfg["val_split"],
                token=True,
            )
        else:
            raise RuntimeError("Either val_percentage or val_split must be specified in the config.")

        test_dataset = load_dataset(
            dataset_params.name,
            split=dataset_params.test_split,
            token=True,
        )

        dataset: DatasetDict = MetadataDatasetDict(
            train=train_dataset,
            val=val_dataset,
            test=test_dataset,
        )

        dataset = preprocess_dataset(dataset, cfg)

        save_dataset_to_disk(dataset, DATASET_DIR)
    else:
        dataset: Dataset = load_from_disk(dataset_path=str(DATASET_DIR))

    return dataset
