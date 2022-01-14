import pytest
import heliq
import numpy as np


def test_accepts_3d_array(aligned_cylinder):
    heliq.helical_axis_pca(aligned_cylinder, 0.5)


def test_returns_array(aligned_cylinder):
    orientation = heliq.helical_axis_pca(aligned_cylinder, 0.5)
    assert type(orientation) is np.ndarray
    assert orientation.ndim == 1
    assert orientation.shape[0] == 3


def test_handles_prealigned_data(aligned_cylinder):
    orientation = heliq.helical_axis_pca(aligned_cylinder, 0.5)
    expected = np.array([0, 0, 1])
    for i in range(3):
        assert orientation[i] == pytest.approx(expected[i])


def test_handles_misaligned_data(aligned_cylinder):
    orientation = heliq.helical_axis_pca(aligned_cylinder.transpose((2, 1, 0)), 0.5)
    expected = np.array([0, 1, 0])
    for i in range(3):
        assert orientation[i] == pytest.approx(expected[i])

    orientation = heliq.helical_axis_pca(aligned_cylinder.transpose((0, 2, 1)), 0.5)
    expected = np.array([1, 0, 0])
    for i in range(3):
        assert orientation[i] == pytest.approx(expected[i])
