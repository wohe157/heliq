"""
This example shows how to calculate and visualize cylindrical sections from
volumetric data in Cartesian coordinates.
"""
import numpy as np
import matplotlib.pyplot as plt
import heliq


# First we need to load the data, see example 1 for more details.
data = np.load("helix.npy")

_, ax = plt.subplots(1, 1)
ax.imshow(data[:, data.shape[1]//2, :].T, cmap='gray', origin='lower')
ax.set_title("Orthoslice of the example data")
ax.set_xlabel("x [voxels]")
ax.set_ylabel("z [voxels]")


# Calculate a cylindrical section using ``heliq.cylindrical_sections``. You can
# calculate multiple sections at once by providing a list to the ``rho``
# argument. If your voxel size is not 1, but e.g. 0.15 nm, and you want to
# calculate a cylindrical section at ρ = 10 nm, you should enter ``rho=10/0.15``
# or in general ``rho=desired_rho/voxel_size``.
data_cyl = heliq.cylindrical_sections(data, rho=80)

_, ax = plt.subplots(1, 1)
ax.imshow(data_cyl, cmap='gray', origin='lower')
ax.set_title("Cylindrical section at ρ = 80 voxels")
ax.set_xlabel("θ [degrees]")
ax.set_ylabel("z [voxels]")


plt.show()
