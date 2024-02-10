from pathlib import Path

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from papers.tasks.image.classification import ImageClassificationTask

SOURCE_URL = (
    "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
)


class PizzaSteakSushiTask(ImageClassificationTask):
    code = "pizza_steak_sushi"
    class_names = ["pizza", "steak", "sushi"]
    train_val_split = (0.9, 0.1)

    def download_data(self, destination: Path):
        self.download_unpack_zip(
            source=SOURCE_URL,
            destination=destination,
        )

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dir = self.cache_directory / "train"
        dataset = datasets.ImageFolder(str(train_dir), transform=self.transform)
        train, _ = random_split(dataset, self.train_val_split)
        return DataLoader(train, batch_size=self.batch_size)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        train_dir = self.cache_directory / "train"
        dataset = datasets.ImageFolder(str(train_dir), transform=self.transform)
        _, val = random_split(dataset, self.train_val_split)
        return DataLoader(val, batch_size=self.batch_size)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dir = self.cache_directory / "test"
        dataset = datasets.ImageFolder(str(test_dir), transform=self.transform)
        return DataLoader(dataset, batch_size=self.batch_size)
