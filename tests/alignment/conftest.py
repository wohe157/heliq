import pytest
import numpy as np


@pytest.fixture
def offset_cylinder():
    """Create a cylinder at a known location"""
    x, y, z = np.meshgrid(np.arange(64), np.arange(64), np.arange(64))
    center = np.array((30, 35, 29))
    data = np.logical_and(
        (x - center[0])**2 + (y - center[1])**2 <= 10**2,
        np.abs(z - center[2]) <= 20).astype(float)
    return center, data


@pytest.fixture
def aligned_cylinder():
    """Create an aligned cylinder at the center"""
    ls = np.linspace(-1, 1, 64)
    x, y, z = np.meshgrid(ls, ls, ls)
    return np.logical_and(x**2 + y**2 <= 0.3**2, np.abs(z) <= 0.6).astype(float)
