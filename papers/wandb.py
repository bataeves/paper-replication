import os
from pathlib import Path
from typing import Union

import wandb
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from wandb.apis.public import Run
from wandb.sdk.lib import RunDisabled

ENV_WANDB_API_KEY = "WANDB_API_KEY"
WANDB_ENTITY = "bataeves"
WANDB_PROJECT = "paper-vit"
WANDB_DEFAULT_DIR = Path("~/.paper/wandb/").expanduser()


def wandb_api_key() -> str | None:
    return os.getenv(ENV_WANDB_API_KEY)


def init(**kwargs) -> Union[Run, RunDisabled, None]:
    api_key = wandb_api_key()
    if not api_key:
        logger.info(
            f"Weights & Biases disabled due to missing API key in {ENV_WANDB_API_KEY}"
        )
        return None
    wandb.login(key=api_key)
    return wandb.init(
        entity=WANDB_ENTITY, project=WANDB_PROJECT, dir=WANDB_DEFAULT_DIR, **kwargs
    )


def get_pl_logger(**kwargs) -> WandbLogger | None:
    experiment = init(**kwargs)
    if experiment is None:
        return None

    return WandbLogger(
        project=WANDB_PROJECT, save_dir=WANDB_DEFAULT_DIR, experiment=experiment
    )
