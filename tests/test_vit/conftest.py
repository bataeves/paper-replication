from pathlib import Path

import pytest
from torch.utils.data import DataLoader
from torchvision import datasets

from papers.vit.data import get_transforms

data_folder = Path(__file__).parent / "data"
IMG_SIZE = 224


@pytest.fixture
def dataloader():
    transforms = get_transforms(IMG_SIZE, IMG_SIZE)
    image_folder = datasets.ImageFolder(str(data_folder), transform=transforms)
    return DataLoader(
        image_folder,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
