from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose

from papers.tasks.image.classification import ImageClassificationTask


class PizzaSteakSushiTask(ImageClassificationTask):
    code = "pizza_steak_sushi"

    def download_data(self, destination: Path):
        self.download_unpack_zip(
            source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
            destination=destination,
        )

    @property
    def class_names(self):
        return ["pizza", "steak", "sushi"]

    @property
    def train_dir(self) -> Path:
        return self.cache_directory / "train"

    @property
    def test_dir(self) -> Path:
        return self.cache_directory / "test"

    def dataloader_train(self, transform: Compose | None = None, **kwargs) -> DataLoader:
        dataset = datasets.ImageFolder(str(self.train_dir), transform=transform)
        return DataLoader(dataset, **kwargs)

    def dataloader_test(self, transform: Compose | None = None, **kwargs) -> DataLoader:
        dataset = datasets.ImageFolder(str(self.test_dir), transform=transform)
        return DataLoader(dataset, **kwargs)
