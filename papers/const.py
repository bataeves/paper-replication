import os
from pathlib import Path

VAR_WANDB_API_KEY = "WANDB_API_KEY"

# DATA_CACHE_DIR = Path.home() / ".paper" / "cache"
DATA_CACHE_DIR = Path(__file__).parent.parent / ".cache"
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def wandb_api_key() -> str | None:
    return os.getenv(VAR_WANDB_API_KEY)
