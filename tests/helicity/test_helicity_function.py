import pytest
import heliq
import numpy as np


@pytest.mark.parametrize("delta_rho", [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
])
def test_handles_negative_delta_rho(delta_rho):
    with pytest.raises(ValueError):
        heliq.helicity_function(np.random.standard_normal((2, 2, 2)), delta_rho=delta_rho)


def test_handles_positive_delta_rho():
    heliq.helicity_function(np.random.standard_normal((2, 2, 2)), delta_rho=3.1)


@pytest.mark.parametrize("delta_alpha", [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
])
def test_handles_negative_delta_alpha(delta_alpha):
    with pytest.raises(ValueError):
        heliq.helicity_function(np.random.standard_normal((2, 2, 2)), delta_alpha=delta_alpha)


def test_returns_helicityfunction_object():
    rvalue = heliq.helicity_function(np.random.standard_normal((2, 2, 2)))
    assert hasattr(rvalue, 'delta_alpha')
    assert hasattr(rvalue, 'delta_rho')
    assert hasattr(rvalue, 'histogram')
