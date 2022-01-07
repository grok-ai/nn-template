import hydra
import omegaconf
from torch.utils.data import Dataset
from torchvision.datasets import FashionMNIST

from nn_template.common.utils import PROJECT_ROOT, Split


class MyDataset(Dataset):
    def __init__(self, split: Split, **kwargs):
        super().__init__()
        self.split: Split = split

        # example
        self.mnist = FashionMNIST(
            kwargs["path"],
            train=split == "train",
            download=True,
            transform=kwargs["transform"],
        )

    def __len__(self) -> int:
        # example
        return len(self.mnist)

    def __getitem__(self, index: int):
        # example
        return self.mnist[index]

    def __repr__(self) -> str:
        return f"MyDataset({self.split=}, n_instances={len(self)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    dataset: MyDataset = hydra.utils.instantiate(
        cfg.nn.data.datasets.train, split="train", _recursive_=False
    )


if __name__ == "__main__":
    main()
