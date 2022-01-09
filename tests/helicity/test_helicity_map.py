import pytest
import heliq
import numpy as np


@pytest.mark.parametrize("sigma", [
    pytest.param(0, id="zero"),
    pytest.param(1, id="positive int"),
    pytest.param(0.3, id="positive float"),
])
def test_handles_positive_sigma(sigma):
    heliq.helicity_map(np.random.standard_normal((2, 2, 2)), sigma)


@pytest.mark.parametrize("threshold", [
    pytest.param(0, id="zero"),
    pytest.param(0.5, id="positive"),
    pytest.param(-0.5, id="negative"),
])
def test_accepts_threshold(threshold):
    heliq.helicity_map(np.random.standard_normal((2, 2, 2)), 0.5, threshold)


def test_rejects_negative_sigma():
    with pytest.raises(ValueError):
        heliq.helicity_map(np.random.standard_normal((2, 2, 2)), -0.5)


@pytest.mark.parametrize("shape", [
    pytest.param((2, 2, 2), id="2,2,2"),
    pytest.param((1, 3, 2), id="1,3,2"),
])
def test_returns_correct_array(shape):
    rvalue = heliq.helicity_map(np.random.standard_normal(shape), 1)
    assert type(rvalue) is np.ndarray
    assert rvalue.ndim == 3
    for i in range(3):
        assert rvalue.shape[i] == shape[i]
