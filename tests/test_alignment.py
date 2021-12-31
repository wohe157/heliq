import unittest
import heliq

import numpy as np


class CenterOfMassTestCase(unittest.TestCase):

    def setUp(self):
        # Create a cylinder at a known location
        x, y, z = np.meshgrid(np.arange(64), np.arange(64), np.arange(64))
        self.center = np.array((30, 35, 29))
        self.data = np.logical_and(
            (x - self.center[0])**2 + (y - self.center[1])**2 <= 10**2,
            np.abs(z - self.center[2]) <= 20).astype(float)

    def test_HandlesValidInputArrays(self):
        """Test that centerOfMass doesn't raise exceptions when provided with correct data"""
        try:
            heliq.centerOfMass(self.data)
        except:
            self.fail("Exception raised when provided with 3D numpy array")

    def test_HandlesInvalidInputArrays(self):
        """Test that centerOfMass does raise exceptions when provided with incorrect data"""
        with self.assertRaises(ValueError, msg="Doesn't complain about 1D data"):
            heliq.centerOfMass(np.zeros((3,)))
        with self.assertRaises(ValueError, msg="Doesn't complain about 2D data"):
            heliq.centerOfMass(np.zeros((3,6)))
        with self.assertRaises(ValueError, msg="Doesn't complain about 4D data"):
            heliq.centerOfMass(np.zeros((3,2,9,1)))

    def test_ReturnsArray(self):
        """Test that centerOfMass returns a numpy array with shape (3,)"""
        center = heliq.centerOfMass(self.data)
        self.assertEqual(np.ndarray, type(center), "Output is not a numpy array")
        self.assertEqual(1, center.ndim, "Output is not a 1D array")
        self.assertEqual(3, center.shape[0], "Output is not a vector with 3 elements")

    def test_CalculatesCorrectCenter(self):
        """Test that the calculated center of mass is correct"""
        center = heliq.centerOfMass(self.data)
        for i in range(3):
            self.assertAlmostEqual(self.center[i], center[i], "Center is not calculated correctly")


class HelicalAxisPcaTestCase(unittest.TestCase):

    def setUp(self):
        # Create an aligned cylinder at the center
        ls = np.linspace(-1, 1, 64)
        x, y, z = np.meshgrid(ls, ls, ls)
        self.data = np.logical_and(x**2 + y**2 <= 0.3**2, np.abs(z) <= 0.6).astype(float)

    def test_HandlesValidInputArrays(self):
        """Test that helicalAxisPca doesn't raise exceptions when provided with correct data"""
        try:
            heliq.helicalAxisPca(self.data, 0.5)
        except:
            self.fail("Exception raised when provided with 3D numpy array")

    def test_HandlesInvalidInputArrays(self):
        """Test that helicalAxisPca does raise exceptions when provided with incorrect data"""
        with self.assertRaises(ValueError, msg="Doesn't complain about 1D data"):
            heliq.helicalAxisPca(np.zeros((3,)), 0.5)
        with self.assertRaises(ValueError, msg="Doesn't complain about 2D data"):
            heliq.helicalAxisPca(np.zeros((3,6)), 0.5)
        with self.assertRaises(ValueError, msg="Doesn't complain about 4D data"):
            heliq.helicalAxisPca(np.zeros((3,2,9,1)), 0.5)

    def test_ReturnsArray(self):
        """Test that helicalAxisPca returns a numpy array with shape (3,)"""
        orientation = heliq.helicalAxisPca(self.data, 0.5)
        self.assertEqual(np.ndarray, type(orientation), "Output is not a numpy array")
        self.assertEqual(1, orientation.ndim, "Output is not a 1D array")
        self.assertEqual(3, orientation.shape[0], "Output is not a vector with 3 elements")

    def test_HandlesPrealignedData(self):
        """Test that the calculated orientation is correct when the object is already aligned with z"""
        orientation = heliq.helicalAxisPca(self.data, 0.5)
        expected = np.array([0, 0, 1])
        for i in range(3):
            self.assertAlmostEqual(expected[i], orientation[i],
                                   "Orientation of prealigned data is not calculated correctly")

    def test_HandlesMisalignedData(self):
        """Test that the calculated orientation is correct when the object is not aligned with z"""
        orientation = heliq.helicalAxisPca(self.data.transpose((2, 1, 0)), 0.5)
        expected = np.array([0, 1, 0])
        for i in range(3):
            self.assertAlmostEqual(expected[i], orientation[i],
                                   "Orientation of data aligned to y-axis is not calculated correctly")

        orientation = heliq.helicalAxisPca(self.data.transpose((0, 2, 1)), 0.5)
        expected = np.array([1, 0, 0])
        for i in range(3):
            self.assertAlmostEqual(expected[i], orientation[i],
                                   "Orientation of data aligned to x-axis is not calculated correctly")


class AlignHelicalAxisTestCase(unittest.TestCase):

    def setUp(self):
        # Create an aligned cylinder at the center
        ls = np.linspace(-1, 1, 64)
        x, y, z = np.meshgrid(ls, ls, ls)
        self.data = np.logical_and(x**2 + y**2 <= 0.3**2, np.abs(z) <= 0.6).astype(float)

    def test_HandlesValidDataOrientationAndCenter(self):
        """Test that alignHelicalAxis doesn't raise exceptions when provided with input arguments"""
        try:
            heliq.alignHelicalAxis(self.data, np.array((0, 0, 1)), np.array((0, 0, 0)))
        except:
            self.fail("Exception raised when provided with 3D numpy array")

    def test_HandlesTupleOrListAsOrientation(self):
        """Test that alignHelicalAxis correctly handles orientation if it is provided as a tuple or list"""
        try:
            heliq.alignHelicalAxis(self.data, (0, 0, 1), np.array((0, 0, 0)))
        except:
            self.fail("Exception raised when provided with tuple")

        try:
            heliq.alignHelicalAxis(self.data, [0, 0, 1], np.array((0, 0, 0)))
        except:
            self.fail("Exception raised when provided with list")

    def test_HandlesTupleOrListAsCenter(self):
        """Test that alignHelicalAxis correctly handles center if it is provided as a tuple or list"""
        try:
            heliq.alignHelicalAxis(self.data, np.array((0, 0, 1)), (0, 0, 0))
        except:
            self.fail("Exception raised when provided with tuple")

        try:
            heliq.alignHelicalAxis(self.data, np.array((0, 0, 1)), [0, 0, 0])
        except:
            self.fail("Exception raised when provided with list")

    def test_HandlesInvalidData(self):
        """Test that alignHelicalAxis does raise exceptions when provided with incorrect data"""
        with self.assertRaises(ValueError, msg="Doesn't complain about 1D data"):
            heliq.alignHelicalAxis(np.zeros((3,)), (0, 0, 1), (0, 0, 0))
        with self.assertRaises(ValueError, msg="Doesn't complain about 2D data"):
            heliq.alignHelicalAxis(np.zeros((3,6)), (0, 0, 1), (0, 0, 0))
        with self.assertRaises(ValueError, msg="Doesn't complain about 4D data"):
            heliq.alignHelicalAxis(np.zeros((3,2,9,1)), (0, 0, 1), (0, 0, 0))

    def test_HandlesInvalidOrientation(self):
        """Test that alignHelicalAxis does raise exceptions when provided with incorrect orientation"""
        with self.assertRaises(ValueError, msg="Doesn't complain about 1D orientation"):
            heliq.alignHelicalAxis(self.data, (1,), (0, 0, 0))
        with self.assertRaises(ValueError, msg="Doesn't complain about 2D orientation"):
            heliq.alignHelicalAxis(self.data, (0, 1), (0, 0, 0))
        with self.assertRaises(ValueError, msg="Doesn't complain about 4D orientation"):
            heliq.alignHelicalAxis(self.data, (0, 0, 1, 0), (0, 0, 0))

    def test_HandlesInvalidCenter(self):
        """Test that alignHelicalAxis does raise exceptions when provided with incorrect center"""
        with self.assertRaises(ValueError, msg="Doesn't complain about 1D orientation"):
            heliq.alignHelicalAxis(self.data, (0, 0, 1), (0,))
        with self.assertRaises(ValueError, msg="Doesn't complain about 2D orientation"):
            heliq.alignHelicalAxis(self.data, (0, 0, 1), (0, 0))
        with self.assertRaises(ValueError, msg="Doesn't complain about 4D orientation"):
            heliq.alignHelicalAxis(self.data, (0, 0, 1), (0, 0, 0, 0))

    def test_RequiresNonzeroOrientation(self):
        """Test that alignHelicalAxis complains when an orientation with all zeros is given"""
        with self.assertRaises(ValueError, msg="Doesn't complain about zero orientation"):
            heliq.alignHelicalAxis(self.data, (0, 0, 0), (0, 0, 0))

    def test_ReturnsSameShapeAsInput(self):
        """Test that alignHelicalAxis returns an array with the same shape and type as the input array"""
        # Don't care about the correct center or orientation in this test
        aligned = heliq.alignHelicalAxis(self.data, (0, 0, 1), (0, 0, 0))
        self.assertEqual(self.data.dtype, aligned.dtype, "Output type is not same as input type")
        for i in range(3):
            self.assertEqual(self.data.shape[i], aligned.shape[i],
                             "Output shape is not same as input shape")
