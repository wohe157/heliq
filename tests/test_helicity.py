import pytest
import heliq

import numpy as np


@pytest.fixture
def hfunc():
    return heliq.HelicityFunction(2, 3, np.random.randn(10, 20))


def test_helicityDescriptor_handlesArray():
    heliq.helicityDescriptor(np.random.randn(2, 2, 2))


@pytest.mark.parametrize("shape", [
    pytest.param((2,), id="1D"),
    pytest.param((2, 2), id="2D"),
    pytest.param((2, 2, 2, 2), id="4D"),
])
def test_helicityDescriptor_rejectsInvalidArray(shape):
    with pytest.raises(ValueError):
        heliq.helicityDescriptor(np.random.randn(*shape))


@pytest.mark.parametrize("data", [
    pytest.param(np.zeros((2, 2, 2)), id="zeros"),
    pytest.param(np.ones((2, 2, 2)), id="ones"),
])
def test_helicityDescriptor_rejectsUniformData(data):
    assert data.min() == data.max()
    with pytest.raises(ValueError):
        heliq.helicityDescriptor(data)


@pytest.mark.parametrize("k", [
    pytest.param(0, id="zero"),
    pytest.param(1, id="one"),
    pytest.param(-1, id="negative"),
    pytest.param(2, id="even"),
])
def test_helicityDescriptor_handlesInvalidKernelSize(k):
    with pytest.raises(ValueError):
        heliq.helicityDescriptor(np.random.randn(2, 2, 2), kernel_size=k)


def test_helicityDescriptor_handlesValidKernelSize():
    heliq.helicityDescriptor(np.random.randn(2, 2, 2), kernel_size=3)


@pytest.mark.parametrize("shape", [
    pytest.param((2, 2, 2), id="2,2,2"),
    pytest.param((1, 3, 2), id="1,3,2"),
])
def test_helicityDescriptor_returns2ArraysSameShape(shape):
    rvalue = heliq.helicityDescriptor(np.random.randn(*shape))
    assert type(rvalue) is tuple
    assert len(rvalue) == 2
    assert len(rvalue[0].shape) == 3
    assert len(rvalue[1].shape) == 3
    for i in range(3):
        assert rvalue[0].shape[i] == shape[i]
        assert rvalue[1].shape[i] == shape[i]


@pytest.mark.parametrize("delta_rho", [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
])
def test_helicityFunction_handlesNegativeDeltaRho(delta_rho):
    with pytest.raises(ValueError):
        heliq.helicityFunction(np.random.randn(2, 2, 2), delta_rho=delta_rho)


def test_helicityFunction_handlesPositiveDeltaRho():
    heliq.helicityFunction(np.random.randn(2, 2, 2), delta_rho=3.1)


@pytest.mark.parametrize("delta_alpha", [
    pytest.param(0, id="zero"),
    pytest.param(-1, id="negative"),
])
def test_helicityFunction_handlesNegativeDeltaAlpha(delta_alpha):
    with pytest.raises(ValueError):
        heliq.helicityFunction(np.random.randn(2, 2, 2), delta_alpha=delta_alpha)


def test_helicityFunction_returnsHelicityFunctionObject():
    rvalue = heliq.helicityFunction(np.random.randn(2, 2, 2))
    assert hasattr(rvalue, 'delta_alpha')
    assert hasattr(rvalue, 'delta_rho')
    assert hasattr(rvalue, 'histogram')


def test_plotHelicityFunction_handlesHelicityFunction(hfunc):
    heliq.plotHelicityFunction(hfunc)


def test_plotHelicityFunction_maxValueIsApplied(hfunc):
    im = heliq.plotHelicityFunction(hfunc, vmax=1)
    vmin, vmax = im.get_clim()
    assert vmin == pytest.approx(-1)
    assert vmax == pytest.approx(1)


@pytest.mark.parametrize("vmax", [
    pytest.param(0, id="zero"),
    pytest.param(-12, id="negative"),
])
def test_plotHelicityFunction_handlesInvalidMaxValue(hfunc, vmax):
    with pytest.raises(ValueError):
        heliq.plotHelicityFunction(hfunc, vmax=vmax)


def test_plotHelicityFunction_correctDataIsUsed(hfunc):
    im = heliq.plotHelicityFunction(hfunc)
    assert np.all(np.abs(hfunc.histogram - im.get_array()) < 1e-7)


def test_totalHelicity_matchesExpectation(hfunc):
    htot = heliq.totalHelicity(hfunc)
    assert htot == pytest.approx(np.sum(hfunc.histogram) *
                                 hfunc.delta_alpha * hfunc.delta_rho)


@pytest.mark.parametrize("sigma", [
    pytest.param(0, id="zero"),
    pytest.param(1, id="positive int"),
    pytest.param(0.3, id="positive float"),
])
def test_helicityMap_handlesPositiveSigma(sigma):
    heliq.helicityMap(np.random.randn(2, 2, 2), sigma)


@pytest.mark.parametrize("threshold", [
    pytest.param(0, id="zero"),
    pytest.param(0.5, id="positive"),
    pytest.param(-0.5, id="negative"),
])
def test_helicityMap_acceptsThreshold(threshold):
    heliq.helicityMap(np.random.randn(2, 2, 2), 0.5, threshold)


def test_helicityMap_rejectsNegativeSigma():
    with pytest.raises(ValueError):
        heliq.helicityMap(np.random.randn(2, 2, 2), -0.5)


@pytest.mark.parametrize("shape", [
    pytest.param((2, 2, 2), id="2,2,2"),
    pytest.param((1, 3, 2), id="1,3,2"),
])
def test_helicityMap_returnsCorrectArray(shape):
    rvalue = heliq.helicityMap(np.random.randn(*shape), 1)
    assert type(rvalue) is np.ndarray
    assert rvalue.ndim == 3
    for i in range(3):
        assert rvalue.shape[i] == shape[i]
