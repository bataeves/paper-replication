import lightning as pl
import pytest


@pytest.fixture(autouse=True)
def seeds():
    pl.seed_everything()
