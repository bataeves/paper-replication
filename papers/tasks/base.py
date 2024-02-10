import os
import zipfile
from pathlib import Path
from typing import Dict, Any

import lightning as pl
import requests
from loguru import logger

DEFAULT_CACHE_DIRECTORY = Path("~/.papers/cache").expanduser()
SUCCESS_FILENAME = "_SUCCESS"


class BaseTask(pl.LightningDataModule):
    code: str
    """
    Machine unique code for the ML task
    """

    def prepare_data(self) -> None:
        directory_path = self.cache_directory
        success_file = directory_path / SUCCESS_FILENAME
        if not success_file.is_file():
            logger.info(f"Success file {success_file} not found, downloading the data")
            self.download_data(directory_path)
            success_file.touch()
        else:
            logger.info(f"Success file found: {success_file}")

    def download_data(self, destination: Path):
        raise NotImplementedError()

    @property
    def cache_directory(self) -> Path:
        directory_path = Path(os.getenv("PAPERS_CACHE_DIR") or DEFAULT_CACHE_DIRECTORY)
        directory_path = directory_path / self.code
        if not directory_path.is_dir():
            logger.info(f"Creating cache directory: {directory_path}")
            directory_path.mkdir(parents=True, exist_ok=True)
        return directory_path

    @staticmethod
    def download_unpack_zip(source: str, destination: Path, remove_source: bool = True):
        """Downloads a zipped dataset from source and unzips to destination.

        Args:
            source (str): A link to a zipped file containing data.
            destination (str): A target directory to unzip data to.
            remove_source (bool): Whether to remove the source after downloading and extracting.

        Returns:
            pathlib.Path to downloaded data.

        Example usage:
            download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                          destination="pizza_steak_sushi")
        """
        if not destination.is_dir():
            logger.info(f"Did not find {destination} directory, creating one...")
            destination.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        target_file = Path(source).name
        temporary_file = destination / target_file
        with open(temporary_file, "wb") as f:
            request = requests.get(source)
            logger.info(f"Downloading {target_file} from {source}...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(temporary_file, "r") as zip_ref:
            logger.info(f"Unzipping {target_file} data to {destination}...")
            zip_ref.extractall(destination)

        # Remove .zip file
        if remove_source:
            os.remove(temporary_file)

    def get_params(self) -> Dict[str, Any]:
        return {
            "code": self.code,
        }
