import pytest
import heliq
import numpy as np


@pytest.fixture
def hfunc():
    return heliq.HelicityFunction(2, 3, np.random.randn(10, 20))
