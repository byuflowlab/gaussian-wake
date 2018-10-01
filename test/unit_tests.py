"""
Created by Jared J. Thomas, Sep. 2018.
FLOW Lab
Brigham Young University
"""

import unittest
import numpy as np

from _porteagel_fortran import porteagel_analyze, porteagel_visualize, x0_func, theta_c_0_func, sigmay_func
from _porteagel_fortran import sigmaz_func, wake_offset_func, deltav_func, deltav_near_wake_lin_func
from _porteagel_fortran import overlap_area_func, wake_combination_func, added_ti_func, k_star_func
from _porteagel_fortran import ct_to_axial_ind_func, wind_shear_func, discontinuity_point_func, smooth_max
from _porteagel_fortran import interpolation, hermite_spline

class test_hermite_spline(unittest.TestCase):

    def test_linear(self):
        """"Approximate y = x - 1"""
        x = 1.
        x0 = 0.
        x1 = 2.
        y0 = -1.
        dy0 = 1.
        y1 = 1.
        dy1 = 1.

        y = hermite_spline(x, x0, x1, y0, dy0, y1, dy1)

        self.assertEqual(y, 0.0)

    def test_cubic(self):
        """Approximate y=x**3"""
        x = 0.
        x0 = -1.
        x1 = 1.
        y0 = 0.
        dy0 = 2.
        y1 = 0.
        dy1 = 2.

        y = hermite_spline(x, x0, x1, y0, dy0, y1, dy1)

        self.assertEqual(y, 0.0)

    def test_parabolic(self):
        """Approximate y=x**2"""
        x = 0.
        x0 = -1.
        x1 = 1.
        y0 = 1.
        dy0 = -2.
        y1 = 1.
        dy1 = 2.

        y = hermite_spline(x, x0, x1, y0, dy0, y1, dy1)

        self.assertEqual(y, 0.0)

class test_interpolation(unittest.TestCase):

    # def test_cubic(self):
    #
    #     # define interpolation type
    #     interp_type = 0
    #
    #     # set up points for interpolation
    #     x = np.array([-1., -0.5, 0., 0.5, 1.])
    #     y = np.array([-1., -0.125, 0., 0.125, 1.])
    #
    #     # set location of interpolation
    #     xval = 0.125
    #
    #     # get interpolated y value
    #     yval = interpolation(interp_type, x, y, xval, 3.0, 3.0, True)
    #
    #     self.assertEqual(yval, 0.0625)

    def test_linear(self):

        # define interpolation type
        interp_type = 1

        # set up points for interpolation
        x = np.array([0., 1., 2.])
        y = np.array([0., 1., 0.])

        # set location of interpolation
        xval = 0.5

        # get interpolated y value
        yval = interpolation(interp_type, x, y, xval, 0.0, 0.0, False)

        self.assertEqual(yval, 0.5)




if __name__ == "__main__":

    unittest.main(verbosity=2)