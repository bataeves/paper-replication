import pytest

from papers.utils.torch import set_seeds


@pytest.fixture(autouse=True)
def seeds():
    set_seeds()
