from collections import namedtuple
from pathlib import Path
from typing import Any, Dict

from anypy.data.metadata_dataset_dict import MetadataDatasetDict
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

from nn_core.common import PROJECT_ROOT

DatasetParams = namedtuple("DatasetParams", ["name", "fine_grained", "train_split", "test_split", "hf_key"])


class HFTransforms:
    def __init__(
        self,
        key: str,
        transforms: Dict,
    ):
        self.transforms = transforms
        self.key = key

    def __call__(self, samples: Dict[str, Any]) -> Dict[str, Any]:
        samples[self.key] = [self.transforms(data) for data in samples[self.key]]
        return samples

    def __repr__(self) -> str:
        return repr(self.transforms)


def preprocess_dataset(
    dataset: Dataset,
    cfg: Dict,
) -> Dataset:
    dataset = dataset.rename_column(cfg["label_key"], cfg["standard_y_key"])
    dataset = dataset.rename_column(cfg["image_key"], cfg["standard_x_key"])

    return dataset


# def add_ids_to_dataset(dataset):
#     N = len(dataset["train"])
#     M = len(dataset["test"])
#     indices = {"train": list(range(N)), "test": list(range(N, N + M))}

#     for mode in ["train", "test"]:
#         dataset[mode] = dataset[mode].map(
#             lambda _, ind: {"id": indices[mode][ind]},
#             with_indices=True,
#             keep_in_memory=True,
#             load_from_cache_file=False,
#         )

#     return dataset


def save_dataset_to_disk(dataset: MetadataDatasetDict, output_path: Path):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True)

    dataset.save_to_disk(output_path)


def load_hf_dataset(**cfg):
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
        # TODO: If adding ids, ensure consistent ids across splits (even random val ones)
        # dataset = add_ids_to_dataset(dataset)

        save_dataset_to_disk(dataset, DATASET_DIR)
    else:
        dataset: Dataset = load_from_disk(dataset_path=str(DATASET_DIR))

    return dataset
