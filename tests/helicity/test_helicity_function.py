import pytest
import heliq
import numpy as np


def test_accepts_positive_delta_rho():
    heliq.helicity_function(np.random.standard_normal((2, 2, 2)), 1., delta_rho=3.1)


def test_accepts_positive_delta_alpha():
    heliq.helicity_function(np.random.standard_normal((2, 2, 2)), 1., delta_alpha=3)


def test_returns_helicityfunction_object():
    rvalue = heliq.helicity_function(np.random.standard_normal((2, 2, 2)), 1.)
    assert hasattr(rvalue, 'delta_alpha')
    assert hasattr(rvalue, 'delta_rho')
    assert hasattr(rvalue, 'histogram')
