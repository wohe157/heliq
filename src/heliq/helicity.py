"""
The helicity of an objct can be analyzed in different ways. The main method
presented in our paper is to calculate the helicity function
(``helicity_function``), which gives the helicity as a function of the distance
between the helical axis and the inclination angle. It is also possible to get
the total helicity (``total_helicity``), which is a single value between -1 and
+1 that indicates the total helicity of an object based on the results of the
helicity function.

If you want to analyze the distribution of helical features across an object,
you can use ``helicity_map``, which returns an array that indicates the local
helicity around each voxel. Use ``helicity_descriptor`` for more controll about
how to calculate this helicity map, this function is internally used by
``helicity_function`` and ``helicity_descriptor`` and returns two arrays: one
array containing the magnitude of helicity (i.e. the gradient magnitude) in
each voxel and one array containing the inclination angle. That way, you can
select for example only the parts of your object that have a certain
inclination angle.
"""
import numpy as np
import dataclasses
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from typing import Tuple


@dataclasses.dataclass
class HelicityFunction:
    """Holds the contents of a helicity function."""
    delta_alpha: float
    delta_rho: float
    histogram: np.ndarray


def helicity_descriptor(data: np.ndarray,
                        kernel_size: int = 5,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the inclination angle and gradient magnitude in each voxel.

    This is not a measure of helicity itself, but is used as the first step in
    most other methods. It can however be used to analyze the helicity of a
    particle with a custom method.

    Args:
        data:
            The voxel data as a 3D array that describes the object.
        kernel_size:
            The size of the Sobel kernel used for calculating the gradient,
            should be an odd integer.

    Returns:
        A tuple of two elements, the first is the normalized gradient magnitude
        in each voxel and the second element is the inclination angle in
        degrees in each voxel.
    """
    if data.ndim != 3:
        raise ValueError(("The data has to be a 3D array, "
                          f"but a {data.ndim}D array was given."))
    if data.max() == data.min():
        raise ValueError(("The data cannot be a uniform array, "
                          "make sure that data.max() != data.min()."))
    if kernel_size <= 1:
        raise ValueError("kernel_size must be greater than 1.")
    if kernel_size % 2 != 1:
        raise ValueError("kernel_size must be an odd integer.")

    x, y, z = np.meshgrid(np.arange(data.shape[1]) - data.shape[1] / 2,
                          np.arange(data.shape[0]) - data.shape[0] / 2,
                          np.arange(data.shape[2]) - data.shape[2] / 2)
    phi = np.arctan2(y, x)

    # The gradient is calculated along cylindrical planes using Sobel filters
    s = np.arange(kernel_size, dtype=float) - kernel_size // 2
    sx, sy, sz = np.meshgrid(s, s, s)
    sr = sx ** 2 + sy ** 2 + sz ** 2
    # prevent division by zero, of which the result should be zero in this case
    sr[sr == 0] = np.Inf

    gx = scipy.ndimage.convolve(data, sx / sr)
    gy = scipy.ndimage.convolve(data, sy / sr)
    gz = scipy.ndimage.convolve(data, sz / sr)
    gphi = - gx * np.sin(phi) + gy * np.cos(phi)

    # gmag should be normalized such that the helicity function is bounded to
    # [-1, +1]
    alpha = np.arctan2(gphi, gz) * 180 / np.pi
    gmag = np.sqrt(gphi ** 2 + gz ** 2)
    gmag /= np.sum(np.abs(gmag))
    return gmag, alpha


def helicity_function(data: np.ndarray,
                      delta_alpha: float = 1.0,
                      delta_rho: float = 1.0,
                      kernel_size: int = 5,
                      ) -> HelicityFunction:
    """Calculate the helicity function for a given object.

    Args:
        data:
            The voxel data as a 3D array that describes the object.
        delta_alpha:
            The bin size for the inclination angles in degrees.
        delta_rho:
            The bin size in the radial direction.
        kernel_size:
            The size of the Sobel kernel used for calculating the gradient,
            should be an odd integer.

    Returns:
        A ``HelicityFunction`` containing the results.
    """
    if delta_alpha <= 0:
        raise ValueError("delta_alpha must be positive.")
    if delta_rho <= 0:
        raise ValueError("delta_rho must be positive.")

    # For binning we need the inclination angle and gradient magnitude, but the
    # gradient magnitude should have the correct sign (+ for right-handed and -
    # for left-handed). The orientation of a feature is orthogonal to the
    # orientation of its gradient, the inclination angle is therefore
    # 90° + atan(gz/gphi) = atan(gphi/gz)
    gmag, alpha = helicity_descriptor(data, kernel_size)
    gmag *= np.sign(alpha)
    alpha = np.abs(alpha)

    x, y, z = np.meshgrid(np.arange(data.shape[1]) - data.shape[1] / 2,
                          np.arange(data.shape[0]) - data.shape[0] / 2,
                          np.arange(data.shape[2]) - data.shape[2] / 2)
    rho = np.sqrt(x ** 2 + y ** 2)

    max_rho = float(min(data.shape[:2]))
    rho_edges_left = np.arange(0., max_rho, delta_rho)  # right = left + delta
    alpha_edges_left = np.arange(0., 90., delta_alpha)
    nbins_rho = len(rho_edges_left)
    nbins_alpha = len(alpha_edges_left)
    bins = np.zeros((nbins_rho, nbins_alpha), dtype=float)

    for i, rho_edge_left in enumerate(rho_edges_left):
        for j, alpha_edge_left in enumerate(alpha_edges_left):
            bins[i, j] = np.sum(gmag[np.logical_and(
                np.logical_and(alpha > alpha_edge_left,
                               alpha <= alpha_edge_left + delta_alpha),
                np.logical_and(rho > rho_edge_left,
                               rho <= rho_edge_left + delta_rho))])

    # The helicity function should be divided by the bin area, which makes it
    # possible to compare intensities between helicity functions with different
    # bin sizes. This does not influence the total helicity, because it takes
    # this into account already in the integration.
    bins /= (delta_rho * delta_alpha)
    return HelicityFunction(delta_alpha, delta_rho, bins)


def plot_helicity_function(hfunc: HelicityFunction,
                           vmax: float = None,
                           axis: plt.Axes = None,
                           cmap: str = 'coolwarm',
                           ) -> AxesImage:
    """Plot the helicity function.

    Args:
        hfunc:
            The helicity function of the object.
        vmax:
            The limits for the intensity will be [-vmax, vmax], by default this
            will be calculated as ``max(abs(helicity_function))``.
        axis:
            The axis on which to draw the helicity function. If not provided,
            the currect active axis will be used.
        cmap:
            The colormap that will be applied, which should ideally be a
            diverging colormap to properly visualize the difference between
            left- and right-handed helicity.

    Returns:
        A matplotlib axis image containing the helicity function.
    """
    if vmax is None:
        vmax = np.max(np.abs(hfunc.histogram))
    if axis is None:
        axis = plt.gca()

    if vmax <= 0:
        raise ValueError("vmax must be positive.")

    im = axis.imshow(hfunc.histogram,
                     interpolation='nearest',
                     origin='lower',
                     extent=(0, 90, 0, hfunc.histogram.shape[0]),
                     cmap=cmap)
    im.set_clim(-vmax, vmax)
    axis.set_aspect('auto')
    return im


def total_helicity(hfunc: HelicityFunction) -> float:
    """Calculate the total helicity of an object.

    The total helicity is a value between +1 and -1 that indicates the total
    helicity of the object: a value close to zero indicates achirality, while a
    value close to +/-1 indicates near-perfect helicity. Positive values
    indicate right-handed helicity and negative values indicate left-handed
    helicity.

    Args:
        hfunc:
            The helicity function of the object.

    Returns:
        The total helicity of the object.
    """
    return np.sum(hfunc.histogram) * hfunc.delta_rho * hfunc.delta_alpha


def helicity_map(data: np.ndarray,
                 sigma: float,
                 threshold: float = None,
                 kernel_size: int = 5,
                 ) -> np.ndarray:
    """Create a helicity map that indicates the helicity around each voxel

    Args:
        data:
            The voxel data (3D array) that describes the object.
        sigma:
            The effective size of a region over which the local helicity will
            be averaged, this is done using Gaussian smoothing in 3D.
        threshold:
            If specified, all voxels in the result where ``data < threshold``
            will be set to zero.
        kernel_size:
            The size of the Sobel kernel used for calculating the gradient,
            should be an odd integer.

    Returns:
        A 3D array with the same shape as ``data`` that contains the 3D
        helicity map.
    """
    if sigma < 0:
        raise ValueError("sigma should be positive.")

    gmag, alpha = helicity_descriptor(data, kernel_size)
    hmap = scipy.ndimage.gaussian_filter(gmag * np.sign(alpha), sigma, mode='constant')

    if threshold is not None:
        hmap[data < threshold] = 0

    return hmap
