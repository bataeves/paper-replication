from pathlib import Path

from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader
from torchvision import datasets

from papers.tasks.image.classification import ImageClassificationTask

SOURCE_URL = (
    "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
)


class PizzaSteakSushiTask(ImageClassificationTask):
    code = "pizza_steak_sushi"

    def download_data(self, destination: Path):
        self.download_unpack_zip(
            source=SOURCE_URL,
            destination=destination,
        )

    @property
    def class_names(self):
        return ["pizza", "steak", "sushi"]

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        train_dir = self.cache_directory / "train"
        dataset = datasets.ImageFolder(str(train_dir), transform=self.transform)
        return DataLoader(dataset)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        test_dir = self.cache_directory / "test"
        dataset = datasets.ImageFolder(str(test_dir), transform=self.transform)
        return DataLoader(dataset)
