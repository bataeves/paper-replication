import importlib
import os
from os.path import dirname, splitext
from typing import List, Iterable, Type, Dict, Any

from loguru import logger
from torchvision.transforms import Compose

from papers.tasks.image.image import ImageTask


class ImageClassificationTask(ImageTask):
    class_names: List[str]

    def __init__(self, batch_size: int = 32, transform: Compose | None = None):
        super().__init__()
        self.transform = transform
        self.batch_size = batch_size

    @classmethod
    def iter_datasets(cls) -> Iterable:
        for filename in os.listdir(dirname(__file__)):
            if not filename.endswith(".py"):
                continue

            module_name = splitext(filename)[0]
            module = importlib.import_module(f"{cls.__module__}.{module_name}")

            for name in dir(module):
                obj = getattr(module, name)
                if isinstance(obj, type) and issubclass(obj, cls) and obj != cls:
                    yield obj

    @classmethod
    def get_task_class(cls, code: str) -> Type["ImageClassificationTask"]:
        known_codes = []
        for ds in cls.iter_datasets():
            if hasattr(ds, "code"):
                if ds.code == code:
                    logger.info(f"Loading task={code}: {ds}")
                    return ds
                known_codes.append(ds.code)

        raise ValueError(
            f"Image classification dataset with {code=} not found. Registered datasets are: {known_codes}"
        )

    def get_params(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_classes": len(self.class_names),
            **super().get_params(),
        }
