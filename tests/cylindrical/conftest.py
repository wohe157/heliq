import pytest
import numpy as np


@pytest.fixture
def volume():
    return np.random.randn(16, 16, 16)
