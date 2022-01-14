import pytest
import heliq
import numpy as np


def test_accepts_helicityfunction(hfunc):
    heliq.plot_helicity_function(hfunc)


def test_applies_vmax(hfunc):
    im = heliq.plot_helicity_function(hfunc, vmax=1)
    vmin, vmax = im.get_clim()
    assert vmin == pytest.approx(-1)
    assert vmax == pytest.approx(1)


def test_uses_correct_data(hfunc):
    im = heliq.plot_helicity_function(hfunc)
    assert np.all(np.abs(hfunc.histogram - im.get_array()) < 1e-7)
