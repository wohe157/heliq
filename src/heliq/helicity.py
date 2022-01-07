"""
The helicity of an objct can be analyzed in different ways. The main method
presented in our paper is to calculate the helicity function
(``helicityFunction``), which gives the helicity as a function of the distance
between the helical axis and the inclination angle. It is also possible to get
the total helicity (``totalHelicity``), which is a single value between -1 and
+1 that indicates the total helicity of an object based on the results of the
helicity function.

If you want to analyze the distribution of helical features across an object,
you can use ``helicityMap``, which returns an array that indicates the local
helicity around each voxel. Use ``helicityDescriptor`` for more controll about
how to calculate this helicity map, this function is internally used by
``helicityFunction`` and ``helicityDescriptor`` and returns two arrays: one
array containing the magnitude of helicity (i.e. the gradient magnitude) in
each voxel and one array containing the inclination angle. That way, you can
select for example only the parts of your object that have a certain
inclination angle.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import AxesImage
from dataclasses import dataclass
from scipy.ndimage import convolve, gaussian_filter
from typing import Tuple


@dataclass
class HelicityFunction:
    """Holds the contents of a helicity function

    Attributes:
        delta_alpha (float)
            The bin size for the inclination angles in degrees.
        delta_rho (float)
            The bin size in the radial direction.
        histogram (numpy array)
            A 2D array containing the helicity function at each bin.
    """
    delta_alpha: float
    delta_rho: float
    histogram: np.ndarray


def helicityDescriptor(data: np.ndarray,
                       kernel_size: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the inclination angle and gradient magnitude in each voxel

    This is not a measure of helicity itself, but is used as the first step in
    most other methods. It can however be used to analyze the helicity of a
    particle with a custom method.

    Arguments:
        data (numpy array)
            The voxel data (3D array) that describes the object.
        kernel_size (int, optional)
            The size of the Sobel kernel used for calculating the gradient,
            should be an odd integer.

    Returns:
        numpy array
            The normalized gradient magnitude in each voxel.
        numpy array
            The inclination angle in degrees in each voxel.
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
    sr[sr == 0] = np.Inf  # prevent division by zero, of which
                          # the result should be zero in this case
    gx = convolve(data, sx / sr)
    gy = convolve(data, sy / sr)
    gz = convolve(data, sz / sr)
    gphi = - gx * np.sin(phi) + gy * np.cos(phi)

    # gmag should be normalized such that the helicity function is bounded to
    # [-1, +1]
    alpha = np.arctan2(gphi, gz) * 180 / np.pi
    gmag = np.sqrt(gphi ** 2 + gz ** 2)
    gmag /= np.sum(np.abs(gmag))
    return gmag, alpha


def helicityFunction(data: np.ndarray,
                     delta_alpha: float = 1.0,
                     delta_rho: float = 1.0,
                     kernel_size: int = 5) -> HelicityFunction:
    """Calculate the helicity function for a given object

    Arguments:
        data (numpy array)
            The voxel data (3D array) that describes the object.
        delta_alpha (float, optional)
            The bin size for the inclination angles in degrees.
        delta_rho (float, optional)
            The bin size in the radial direction.
        kernel_size (int, optional)
            The size of the Sobel kernel used for calculating the gradient,
            should be an odd integer.

    Returns:
        HelicityFunction
            This is a container class that holds all the information required
            to analyze the results.
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
    gmag, alpha = helicityDescriptor(data, kernel_size)
    gmag *= np.sign(alpha)
    alpha = np.abs(alpha)

    # Start binning
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


def plotHelicityFunction(helicity_function: HelicityFunction,
                         vmax: float = None,
                         axis: plt.Axes = None,
                         cmap: str = 'coolwarm') -> AxesImage:
    """Plot the helicity function

    Arguments:
        helicity_function (HelicityFunction)
            The helicity function of the object.
        vmax (float, optional)
            The limits for the intensity will be [-vmax, vmax], by default this
            will be calculated as ``max(abs(helicity_function))``.
        axis (matplotlib axis, optional)
            The axis on which to draw the helicity function. If not provided,
            the currect active axis will be used.
        cmap (string, optional)
            The colormap that will be applied, which should ideally be a
            diverging colormap to properly visualize the difference between
            left- and right-handed helicity.

    Returns:
        matplotlib axis image
            The image containing the helicity function
    """
    if vmax is None:
        vmax = np.max(np.abs(helicity_function.histogram))
    if axis is None:
        axis = plt.gca()

    if vmax <= 0:
        raise ValueError("vmax must be positive.")

    im = axis.imshow(helicity_function.histogram,
                     interpolation='nearest',
                     origin='lower',
                     extent=(0, 90, 0, helicity_function.histogram.shape[0]),
                     cmap=cmap)
    im.set_clim(-vmax, vmax)
    axis.set_aspect('auto')
    return im


def totalHelicity(helicity_function: HelicityFunction) -> float:
    """Calculate the total helicity of an object

    The total helicity is a value between +1 and -1 that indicates the total
    helicity of the object: a value close to zero indicates achirality, while a
    value close to +/-1 indicates near-perfect helicity. Positive values
    indicate right-handed helicity and negative values indicate left-handed
    helicity.

    Arguments:
        helicity_function (HelicityFunction)
            The helicity function of the object.

    Returns:
        float
            The total helicity of the object.
    """
    return np.sum(helicity_function.histogram) * \
        helicity_function.delta_rho * helicity_function.delta_alpha


def helicityMap(data: np.ndarray,
                sigma: float,
                threshold: float = None,
                kernel_size: int = 5) -> np.ndarray:
    """Create a helicity map that indicates the helicity around each voxel

    Arguments:
        data (numpy array)
            The voxel data (3D array) that describes the object.
        sigma (float)
            The effective size of a region over which the local helicity will
            be averaged, this is done using Gaussian smoothing in 3D.
        threshold (float, optional)
            If specified, all voxels in the result where ``data < threshold``
            will be set to zero.
        kernel_size (int, optional)
            The size of the Sobel kernel used for calculating the gradient,
            should be an odd integer.

    Returns:
        numpy array
            A 3D array with the same shape as ``data`` that contains the 3D
            helicity map.
    """
    if sigma < 0:
        raise ValueError("sigma should be positive.")

    gmag, alpha = helicityDescriptor(data, kernel_size)
    hmap = gaussian_filter(gmag * np.sign(alpha), sigma, mode='constant')

    if threshold is not None:
        hmap[data < threshold] = 0

    return hmap
