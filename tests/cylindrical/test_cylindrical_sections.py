import pytest
import heliq
import numpy as np


def test_accepts_3d_array(volume):
    heliq.cylindrical_sections(volume)


@pytest.mark.parametrize("rho", [
    pytest.param(5, id="scalar"),
    pytest.param([5, 6, 7], id="list"),
    pytest.param((5, 6, 7), id="tuple"),
    pytest.param(np.array((5, 6, 7)), id="array"),
])
def test_accepts_valid_rho(volume, rho):
    heliq.cylindrical_sections(volume, rho=rho)


def test_accepts_positive_ntheta(volume):
    heliq.cylindrical_sections(volume, n_theta=3)


@pytest.mark.parametrize("n_theta", [
    pytest.param(0, id="zero"),
    pytest.param(-2, id="negative"),
])
def test_rejects_negative_ntheta(volume, n_theta):
    with pytest.raises(AssertionError):
        heliq.cylindrical_sections(volume, n_theta=n_theta)


def test_returns_correct_shape(volume):
    cyl = heliq.cylindrical_sections(volume, rho=3, n_theta=5)
    assert cyl.shape == (volume.shape[2], 5, 1)

    cyl = heliq.cylindrical_sections(volume, rho=(3, 4, 5), n_theta=5)
    assert cyl.shape == (volume.shape[2], 5, 3)
