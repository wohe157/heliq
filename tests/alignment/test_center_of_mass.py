import pytest
import heliq
import numpy as np


def test_accepts_3d_array(offset_cylinder):
    heliq.center_of_mass(offset_cylinder[1])


@pytest.mark.parametrize("shape", [
    pytest.param((3,), id="1D"),
    pytest.param((3, 6), id="2D"),
    pytest.param((2, 2, 9, 1), id="4D"),
])
def test_rejects_invalid_nd_array(shape):
    with pytest.raises(ValueError):
        heliq.center_of_mass(np.zeros(shape))


def test_returns_array(offset_cylinder):
    center = heliq.center_of_mass(offset_cylinder[1])
    assert type(center) is np.ndarray
    assert center.ndim == 1
    assert center.shape[0] == 3


def test_calculates_correct_center(offset_cylinder):
    center = heliq.center_of_mass(offset_cylinder[1])
    for i in range(3):
        assert center[i] == offset_cylinder[0][i]
