import pytest
import heliq
import numpy as np


def test_accepts_3d_array(aligned_cylinder):
    heliq.align_helical_axis(aligned_cylinder, (0, 0, 1), (0, 0, 0))


@pytest.mark.parametrize("shape", [
    pytest.param((3,), id="1D"),
    pytest.param((3, 6), id="2D"),
    pytest.param((2, 2, 9, 1), id="4D"),
])
def test_rejects_invalid_nd_array(shape):
    with pytest.raises(ValueError):
        heliq.align_helical_axis(np.zeros(shape), (0, 0, 1), (0, 0, 0))


@pytest.mark.parametrize("orientation,center", [
    pytest.param(np.array((0, 0, 1)), np.array((0, 0, 0)), id="arrays"),
    pytest.param((0, 0, 1), (0, 0, 0), id="tuples"),
    pytest.param([0, 0, 1], [0, 0, 0], id="lists"),
])
def test_accepts_arrays_lists_tuples(aligned_cylinder, orientation, center):
    heliq.align_helical_axis(aligned_cylinder, orientation, center)


@pytest.mark.parametrize("orientation", [
    pytest.param((1,), id="1D"),
    pytest.param((0, 1), id="2D"),
    pytest.param((0, 0, 1, 0), id="4D"),
])
def test_rejects_invalid_orientation_dim(aligned_cylinder, orientation):
    with pytest.raises(ValueError):
        heliq.align_helical_axis(aligned_cylinder, orientation, (0, 0, 0))


@pytest.mark.parametrize("center", [
    pytest.param((0,), id="1D"),
    pytest.param((0, 0), id="2D"),
    pytest.param((0, 0, 0, 0), id="4D"),
])
def test_rejects_invalid_center_dim(aligned_cylinder, center):
    with pytest.raises(ValueError):
        heliq.align_helical_axis(aligned_cylinder, (0, 0, 1), center)


def test_rejects_orientation_onlyzero(aligned_cylinder):
    with pytest.raises(ValueError):
        heliq.align_helical_axis(aligned_cylinder, (0, 0, 0), (0, 0, 0))


def test_returns_same_shape_as_input(aligned_cylinder):
    aligned = heliq.align_helical_axis(aligned_cylinder, (0, 0, 1), (0, 0, 0))
    assert aligned.dtype is aligned_cylinder.dtype
    for i in range(3):
        assert aligned.shape[i] == aligned_cylinder.shape[i]


@pytest.mark.parametrize("orientation", [
    pytest.param((0, 0, 1), id="prealigned"),
    pytest.param((0, 1, 1), id="misaligned"),
])
def test_doesnt_change_volume(aligned_cylinder, orientation):
    center = [d / 2 for d in aligned_cylinder.shape]
    aligned = heliq.align_helical_axis(aligned_cylinder, orientation, center)
    # Volume is always going to change due to interpolation, just make sure
    # that the object doesn't (partly) dissapear
    assert np.sum(aligned) == pytest.approx(np.sum(aligned_cylinder), rel=1e-2)
