import pytest
import heliq

import numpy as np


@pytest.fixture
def offsetCylinder():
    """Create a cylinder at a known location"""
    x, y, z = np.meshgrid(np.arange(64), np.arange(64), np.arange(64))
    center = np.array((30, 35, 29))
    data = np.logical_and(
        (x - center[0])**2 + (y - center[1])**2 <= 10**2,
        np.abs(z - center[2]) <= 20).astype(float)
    return center, data


@pytest.fixture
def alignedCylinder():
    """Create an aligned cylinder at the center"""
    ls = np.linspace(-1, 1, 64)
    x, y, z = np.meshgrid(ls, ls, ls)
    return np.logical_and(x**2 + y**2 <= 0.3**2, np.abs(z) <= 0.6).astype(float)


def test_centerOfMass_handlesValidInputArrays(offsetCylinder):
    heliq.centerOfMass(offsetCylinder[1])


@pytest.mark.parametrize("shape", [
    pytest.param((3,), id="1D"),
    pytest.param((3, 6), id="2D"),
    pytest.param((2, 2, 9, 1), id="4D"),
])
def test_centerOfMass_handlesInvalidInputArrays(shape):
    with pytest.raises(ValueError):
        heliq.centerOfMass(np.zeros(shape))


def test_centerOfMass_returnsArray(offsetCylinder):
    center = heliq.centerOfMass(offsetCylinder[1])
    assert type(center) is np.ndarray
    assert center.ndim == 1
    assert center.shape[0] == 3


def test_centerOfMass_calculatesCorrectCenter(offsetCylinder):
    center = heliq.centerOfMass(offsetCylinder[1])
    for i in range(3):
        assert center[i] == offsetCylinder[0][i]


def test_helicalAxisPca_handlesValidInputArrays(alignedCylinder):
    heliq.helicalAxisPca(alignedCylinder, 0.5)


@pytest.mark.parametrize("shape", [
    pytest.param((3,), id="1D"),
    pytest.param((3, 6), id="2D"),
    pytest.param((2, 2, 9, 1), id="4D"),
])
def test_helicalAxisPca_handlesInvalidInputArrays(shape):
    with pytest.raises(ValueError):
        heliq.helicalAxisPca(np.zeros(shape), 0.5)


def test_helicalAxisPca_returnsArray(alignedCylinder):
    orientation = heliq.helicalAxisPca(alignedCylinder, 0.5)
    assert type(orientation) is np.ndarray
    assert orientation.ndim == 1
    assert orientation.shape[0] == 3


def test_helicalAxisPca_handlesPrealignedData(alignedCylinder):
    orientation = heliq.helicalAxisPca(alignedCylinder, 0.5)
    expected = np.array([0, 0, 1])
    for i in range(3):
        assert orientation[i] == pytest.approx(expected[i])


def test_helicalAxisPca_handlesMisalignedData(alignedCylinder):
    orientation = heliq.helicalAxisPca(alignedCylinder.transpose((2, 1, 0)), 0.5)
    expected = np.array([0, 1, 0])
    for i in range(3):
        assert orientation[i] == pytest.approx(expected[i])

    orientation = heliq.helicalAxisPca(alignedCylinder.transpose((0, 2, 1)), 0.5)
    expected = np.array([1, 0, 0])
    for i in range(3):
        assert orientation[i] == pytest.approx(expected[i])


@pytest.mark.parametrize("orientation,center", [
    pytest.param(np.array((0, 0, 1)), np.array((0, 0, 0)), id="arrays"),
    pytest.param((0, 0, 1), (0, 0, 0), id="tuples"),
    pytest.param([0, 0, 1], [0, 0, 0], id="lists"),
    pytest.param((0, 1, 0), (0, 0, 0), id="misaligned"),
])
def test_alignHelicalAxis_handlesValidArgs(alignedCylinder, orientation, center):
    heliq.alignHelicalAxis(alignedCylinder, orientation, center)


@pytest.mark.parametrize("shape", [
    pytest.param((3,), id="1D"),
    pytest.param((3, 6), id="2D"),
    pytest.param((2, 2, 9, 1), id="4D"),
])
def test_alignHelicalAxis_handlesInvalidData(shape):
    with pytest.raises(ValueError):
        heliq.alignHelicalAxis(np.zeros(shape), (0, 0, 1), (0, 0, 0))


@pytest.mark.parametrize("orientation", [
    pytest.param((1,), id="1D"),
    pytest.param((0, 1), id="2D"),
    pytest.param((0, 0, 1, 0), id="4D"),
])
def test_alignHelicalAxis_handlesInvalidOrientation(alignedCylinder, orientation):
    with pytest.raises(ValueError):
        heliq.alignHelicalAxis(alignedCylinder, orientation, (0, 0, 0))


@pytest.mark.parametrize("center", [
    pytest.param((0,), id="1D"),
    pytest.param((0, 0), id="2D"),
    pytest.param((0, 0, 0, 0), id="4D"),
])
def test_alignHelicalAxis_handlesInvalidCenter(alignedCylinder, center):
    with pytest.raises(ValueError):
        heliq.alignHelicalAxis(alignedCylinder, (0, 0, 1), center)


def test_alignHelicalAxis_requiresNonzeroOrientation(alignedCylinder):
    with pytest.raises(ValueError):
        heliq.alignHelicalAxis(alignedCylinder, (0, 0, 0), (0, 0, 0))


def test_alignHelicalAxis_returnsSameShapeAsInput(alignedCylinder):
    # Don't care about the correct center or orientation in this test
    aligned = heliq.alignHelicalAxis(alignedCylinder, (0, 0, 1), (0, 0, 0))
    assert aligned.dtype is alignedCylinder.dtype
    for i in range(3):
        assert aligned.shape[i] == alignedCylinder.shape[i]


@pytest.mark.parametrize("orientation", [
    pytest.param((0, 0, 1), id="prealigned"),
    pytest.param((0, 1, 1), id="misaligned"),
])
def test_alignHelicalAxis_doesntChangeVolume(alignedCylinder, orientation):
    center = [d / 2 for d in alignedCylinder.shape]
    aligned = heliq.alignHelicalAxis(alignedCylinder, orientation, center)
    # Volume is always going to change due to interpolation, just make sure
    # that the object doesn't (partly) dissapear
    assert np.sum(aligned) == pytest.approx(np.sum(alignedCylinder), rel=1e-2)
