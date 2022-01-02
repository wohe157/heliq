"""
To calculate the helicity of a certain object, it is important to make sure
that the helical axis is aligned to the z-axis and is centered in the volume.
This can be done by finding the center and orientation of your object and
entering that information in ``alignHelicalAxis``. That function will rotate
and translate your data such that the given orientation becomes parallel with
the z-axis and is centered in the volume. You can find the orientation of the
helical axis and the center either manually using external 3D visualization
software, or for relatively simple objects these can also be found
automatically using ``centerOfMass`` and/or ``helicalAxisPca``. The former
assumes that the helical axis goes through the center of mass of you object and
will therefore select the center of mass as the center of the object. The
latter assumes that the object is longer in one direction compared to the other
directions, and will use this direction as the helical axis. Don't use these
functions if these assumptions are not correct for your object!
"""
import numpy as np
from scipy.ndimage import center_of_mass, map_coordinates
from scipy.spatial.transform import Rotation


def centerOfMass(data: np.ndarray) -> np.ndarray:
    """Calculate the center of mass of a 3D object

    Arguments:
        data (numpy array)
            A 3D array containing the shape of an object. The shape of the
            array should correspond to the (y, x, z) axes.

    Returns:
        numpy array
            A vector of 3 elements (x, y, z) of the center of mass of the
            object.
    """
    if data.ndim != 3:
        raise ValueError(("Expected a 3D input array, "
                          f"but got {data.ndim:d} dimensions."))

    return np.asarray(center_of_mass(data))[[1, 0, 2]]


def helicalAxisPca(data: np.ndarray, threshold: float) -> np.ndarray:
    """Estimate the orientation of the helical axis of a 3D object using PCA

    The helical axis is assumed to be along the longest direction of the
    object. This is often true for rods, wires or similar shapes, but is not
    guaranteed to be the case. The longest direction can be detected using
    principal component analysis (PCA) by binarizing it using a certain
    threshold value. A list will then be made with the (x, y, z) coordinates of
    all the voxels that are considered part of the object, i.e. all the voxels
    with an intensity that is larger than the threshold value. PCA is executed
    on this list of coordinates such that the component with the largest score
    will be oriented in the direction along which the points are spread the
    most. This is the direction along which the object is the longest.

    Arguments:
        data (numpy array)
            A 3D array containing the shape of an object. The shape of the
            array should correspond to the (y, x, z) axes.
        threshold (float)
            The threshold that will be used to binarize the data. Only voxels
            with a value larger than this threshold will be used.

    Returns:
        numpy array
            A vector of 3 elements (x, y, z) of the orientation of the helical
            axis.
    """
    if data.ndim != 3:
        raise ValueError(("Expected a 3D input array, "
                          f"but got {data.ndim:d} dimensions."))

    points = np.argwhere(data > threshold)[:, (1, 0, 2)]  # (y,x,z) to (x,y,z)
    covariancematrix = np.cov(points.T)
    eigval, eigvec = np.linalg.eig(covariancematrix)

    orientation = eigvec[:, np.argmax(eigval)]
    return orientation / np.linalg.norm(orientation)


def alignHelicalAxis(data: np.ndarray, orientation: np.ndarray,
                     center: np.ndarray) -> np.ndarray:
    """Transform a 3D object to align its helical axis to the z-axis

    Arguments:
        data (numpy array)
            A 3D array containing the shape of a (helical) object. The shape
            of the array should correspond to the (y, x, z) axes.
        orientation (numpy array or tuple or list)
            A vector that points in the direction of the helical axis. The
            vector should be a numpy array, tuple or list with three components
            (x, y, z).
        center (numpy array or tuple or list)
            The location of the center of the object represented as a nnumpy
            array, tuple or list with components (x, y, z). This is typically
            the center of mass of the object.

    Returns:
        numpy array
            The transformed data. The helical axis of this object is parallel
            with the z-axis and is positioned in the center of the array in
            each dimension.
    """
    orientation = np.asarray(orientation)
    center = np.asarray(center)
    if data.ndim != 3:
        raise ValueError(("Expected data to be a 3D input array, "
                          f"but got {data.ndim:d} dimensions."))
    if orientation.ndim != 1 or orientation.shape[0] != 3:
        raise ValueError(("Expected orientation to have shape (3,), "
                          f"but got {orientation.shape}."))
    if center.ndim != 1 or center.shape[0] != 3:
        raise ValueError(("Expected center to have shape (3,), "
                          f"but got {center.shape}."))
    if np.linalg.norm(orientation) < 1e-7:
        raise ValueError("The orientation must contain nonzero elements.")

    ymax, xmax, zmax = data.shape
    x, y, z = np.meshgrid(np.arange(xmax, dtype=float),
                          np.arange(ymax, dtype=float),
                          np.arange(zmax, dtype=float))

    # Translate such that origin is at middle of array
    x -= xmax / 2
    y -= ymax / 2
    z -= zmax / 2

    # Rotate such that orientation is aligned with z-axis
    orientation = orientation / np.linalg.norm(orientation)
    zaxis = np.array((0, 0, 1))

    angle = np.arccos(np.dot(zaxis, orientation))
    if angle > 1e-7:
        rvec = np.cross(zaxis, orientation)
        rvec = rvec * angle / np.linalg.norm(rvec)
        rmat = Rotation.from_rotvec(rvec).as_matrix()

        xr = rmat[0, 0] * x + rmat[0, 1] * y + rmat[0, 2] * z
        yr = rmat[1, 0] * x + rmat[1, 1] * y + rmat[1, 2] * z
        zr = rmat[2, 0] * x + rmat[2, 1] * y + rmat[2, 2] * z

        x = xr
        y = yr
        z = zr

    # Translate such that center is at origin
    x += center[0]
    y += center[1]
    z += center[2]

    # Interpolate data
    return map_coordinates(data, np.stack((y, x, z), axis=0))
