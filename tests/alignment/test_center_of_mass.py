import heliq
import numpy as np


def test_accepts_3d_array(offset_cylinder):
    heliq.center_of_mass(offset_cylinder[1])


def test_returns_array(offset_cylinder):
    center = heliq.center_of_mass(offset_cylinder[1])
    assert type(center) is np.ndarray
    assert center.ndim == 1
    assert center.shape[0] == 3


def test_calculates_correct_center(offset_cylinder):
    center = heliq.center_of_mass(offset_cylinder[1])
    for i in range(3):
        assert center[i] == offset_cylinder[0][i]
