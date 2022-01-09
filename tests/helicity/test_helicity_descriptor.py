import pytest
import heliq
import numpy as np


def test_accepts_3d_array():
    heliq.helicity_descriptor(np.random.standard_normal((2, 2, 2)))


@pytest.mark.parametrize("shape", [
    pytest.param((2,), id="1D"),
    pytest.param((2, 2), id="2D"),
    pytest.param((2, 2, 2, 2), id="4D"),
])
def test_rejects_invalid_nd_array(shape):
    with pytest.raises(ValueError):
        heliq.helicity_descriptor(np.random.standard_normal(shape))


@pytest.mark.parametrize("data", [
    pytest.param(np.zeros((2, 2, 2)), id="zeros"),
    pytest.param(np.ones((2, 2, 2)), id="ones"),
])
def test_rejects_uniform_data(data):
    assert data.min() == data.max()
    with pytest.raises(ValueError):
        heliq.helicity_descriptor(data)


@pytest.mark.parametrize("k", [
    pytest.param(3, id="zero"),
    pytest.param(5, id="one"),
])
def test_accepts_valid_kernel_size(k):
    heliq.helicity_descriptor(np.random.standard_normal((2, 2, 2)), kernel_size=k)


@pytest.mark.parametrize("k", [
    pytest.param(0, id="zero"),
    pytest.param(1, id="one"),
    pytest.param(-1, id="negative"),
    pytest.param(2, id="even"),
])
def test_reject_invalid_kernel_size(k):
    with pytest.raises(ValueError):
        heliq.helicity_descriptor(np.random.standard_normal((2, 2, 2)), kernel_size=k)


@pytest.mark.parametrize("shape", [
    pytest.param((2, 2, 2), id="2,2,2"),
    pytest.param((1, 3, 2), id="1,3,2"),
])
def test_returns_two_arrays_same_shape(shape):
    rvalue = heliq.helicity_descriptor(np.random.standard_normal(shape))
    assert type(rvalue) is tuple
    assert len(rvalue) == 2
    assert len(rvalue[0].shape) == 3
    assert len(rvalue[1].shape) == 3
    for i in range(3):
        assert rvalue[0].shape[i] == shape[i]
        assert rvalue[1].shape[i] == shape[i]
