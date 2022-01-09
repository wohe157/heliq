import pytest
import heliq
import numpy as np


def test_matches_expectation(hfunc):
    htot = heliq.total_helicity(hfunc)
    assert htot == pytest.approx(np.sum(hfunc.histogram) * hfunc.delta_alpha * hfunc.delta_rho)
