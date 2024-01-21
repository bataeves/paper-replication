import os

VAR_WANDB_API_KEY = "WANDB_API_KEY"


def wandb_api_key() -> str | None:
    return os.getenv(VAR_WANDB_API_KEY)
