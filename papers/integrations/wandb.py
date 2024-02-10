import os
from pathlib import Path
from typing import Union, Optional

import wandb
from lightning.pytorch.loggers import WandbLogger
from loguru import logger
from wandb.apis.public import Run
from wandb.sdk.lib import RunDisabled

ENV_WANDB_API_KEY = "WANDB_API_KEY"
ENV_WANDB_PROJECT = "WANDB_PROJECT"
ENV_WANDB_ENTITY = "WANDB_ENTITY"
WANDB_ENTITY = "bataeves"
WANDB_PROJECT = "paper-vit"
WANDB_DEFAULT_DIR = Path("~/.paper/wandb/").expanduser()


def wandb_api_key() -> str | None:
    return os.getenv(ENV_WANDB_API_KEY)


def wandb_entity() -> str:
    return os.getenv(ENV_WANDB_ENTITY) or WANDB_ENTITY


def wandb_project() -> str:
    return os.getenv(ENV_WANDB_PROJECT) or WANDB_PROJECT


def init(group: Optional[str] = None, **kwargs) -> Union[Run, RunDisabled, None]:
    api_key = wandb_api_key()
    if not api_key:
        logger.info(
            f"Weights & Biases disabled due to missing API key in {ENV_WANDB_API_KEY}"
        )
        return None
    wandb.login(key=api_key)
    return wandb.init(
        entity=WANDB_ENTITY,
        project=wandb_project(),
        dir=WANDB_DEFAULT_DIR,
        group=group,
        **kwargs,
    )


def get_pl_logger(
    experiment: Optional[Union["Run", "RunDisabled"]] = None, **kwargs
) -> WandbLogger | None:
    if experiment is None:
        experiment = init()
        if experiment is None:
            return None

    return WandbLogger(
        project=wandb_project(), save_dir=WANDB_DEFAULT_DIR, experiment=experiment, **kwargs
    )
