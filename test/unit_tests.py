"""
Created by Jared J. Thomas, Sep. 2018.
FLOW Lab
Brigham Young University
"""

import unittest
import numpy as np

from _porteagel_fortran import porteagel_analyze, porteagel_visualize, x0_func, theta_c_0_func, sigmay_func, sigma_spread_func
from _porteagel_fortran import sigmaz_func, wake_offset_func, deltav_func, deltav_near_wake_lin_func
from _porteagel_fortran import overlap_area_func, wake_combination_func, added_ti_func, k_star_func
from _porteagel_fortran import ct_to_axial_ind_func, wind_shear_func, discontinuity_point_func, smooth_max
from _porteagel_fortran import interpolation, hermite_spline
from openmdao.api import Problem, Group

class test_basic_subroutines(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1E-6
        self.d = 126.4
        self.yaw = np.pi/6.
        self.ct = 0.8
        self.alpha = 2.32
        self.beta = 0.154
        self.ti = 0.1
        self.ky = 0.25
        self.kz = 0.2

    def test_x0_func(self):

        x0 = x0_func(self.d, self.yaw, self.ct, self.alpha, self.ti, self.beta)

        self.assertAlmostEqual(x0, 353.2313474, delta=self.tolerance)

    def test_discontinuity_point_func(self):

        x0 = 353.0

        xd = discontinuity_point_func(x0, self.d, self.ky, self.kz, self.yaw, self.ct)

        self.assertAlmostEqual(xd, 335.5180515, delta=self.tolerance)

    def test_sigmay_func(self):

        deltax0 = 500. - 353.0

        xd = sigmay_func(self.ky, deltax0, self.d, self.yaw)

        self.assertAlmostEqual(xd, 75.45193794, delta=self.tolerance)

    def test_sigmaz_func(self):

        deltax0 = 500. - 353.0

        xd = sigmaz_func(self.kz, deltax0, self.d)

        self.assertAlmostEqual(xd, 74.08914857, delta=self.tolerance)

    def test_theta_c_0_func(self):

        theta_c_0 = theta_c_0_func(self.yaw, self.ct)

        self.assertAlmostEqual(theta_c_0, , delta=self.tolerance)

class test_sigma_spread(unittest.TestCase):

    def setUp(self):
        self.tolerance = 1E-6
        self.d = 126.4
        self.yaw = np.pi / 6.
        self.ct = 0.8
        self.alpha = 2.32
        self.beta = 0.154
        self.ti = 0.1
        self.ky = 0.25
        self.kz = 0.2

        x = np.array([500.0, 500.0, 500.0, 200.0, -10.0])
        xi_d = np.array([1.0, 2.0, 1.0, 1.0, 1.0])
        xi_a = np.array([1.0, 1.0, 50.0, 1.0, 1.0])

        sigma_0 = 38.7
        sigma_d = 4.14

        x0 = 353.0

        self.sigma_spread = np.zeros_like(x)
        for i in np.arange(0, x.size):
            self.sigma_spread[i] = sigma_spread_func(x[i], xi_a[i], xi_d[i], self.ky, x0, sigma_0, sigma_d)

        self.correct_results = np.array([75.45, 150.9, 2451.73206799, 23.72073654, 0.0])

    def test_sigma_spread_func_case1(self):

        self.assertAlmostEqual(self.sigma_spread[0], self.correct_results[0], delta=self.tolerance)

    def test_sigma_spread_func_case2(self):

        self.assertAlmostEqual(self.sigma_spread[1], self.correct_results[1], delta=self.tolerance)

    def test_sigma_spread_func_case3(self):

        self.assertAlmostEqual(self.sigma_spread[2], self.correct_results[2], delta=self.tolerance)

    def test_sigma_spread_func_case4(self):

        self.assertAlmostEqual(self.sigma_spread[3], self.correct_results[3], delta=self.tolerance)

    def test_sigma_spread_func_case5(self):

        self.assertAlmostEqual(self.sigma_spread[4], self.correct_results[4], delta=self.tolerance)



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

class test_ctcp_curve(unittest.TestCase):

    def setUp(self):
        filename = "./input_files/NREL5MWCPCT_dict.p"
        import cPickle as pickle
        data = pickle.load(open(filename, "rb"))
        cp_data = np.zeros([data['wind_speed'].size])
        ct_data = np.zeros([data['wind_speed'].size])
        wind_speed_data = np.zeros([data['wind_speed'].size])
        cp_data[:] = data['CP']
        ct_data[:] = data['CT']
        wind_speed_data[:] = data['wind_speed']

        self.ct_data = ct_data
        self.cp_data = cp_data
        self.wind_speed_data = wind_speed_data

        self.options = {'use_ct_curve': True,
                   'ct_curve_ct': self.ct_data,
                   'ct_curve_wind_speed': self.wind_speed_data}

    def test_5mw_ct_greater_than_1_warning(self):

        from gaussianwake.gaussianwake import GaussianWake
        import pytest

        pytest.warns(Warning, GaussianWake, nTurbines=6, options=self.options)

class test_wec(unittest.TestCase):

    def setUp(self):
        filename = "./input_files/NREL5MWCPCT_dict.p"
        import cPickle as pickle
        data = pickle.load(open(filename, "rb"))
        cp_data = np.zeros([data['wind_speed'].size])
        ct_data = np.zeros([data['wind_speed'].size])
        wind_speed_data = np.zeros([data['wind_speed'].size])
        cp_data[:] = data['CP']
        ct_data[:] = data['CT']
        wind_speed_data[:] = data['wind_speed']

        self.ct_data = ct_data
        self.cp_data = cp_data
        self.wind_speed_data = wind_speed_data

        self.options = {'use_ct_curve': True,
                   'ct_curve_ct': self.ct_data,
                   'ct_curve_wind_speed': self.wind_speed_data}

        nTurbines = 2
        from gaussianwake.gaussianwake import GaussianWake
        prob = Problem(root=Group())
        prob.root.add('wakemodel', GaussianWake(nTurbines, options=self.options), promotes=['*'])
        prob.setup()
        prob['wind_speed'] = 8.
        self.prob = prob

    def test_no_change_in_deficit_by_wake_spread_rate_multiplier_at_center(self):
        prob = self.prob
        turbineX = np.array([0., 400.])
        turbineY = np.array([0., 0.])
        rotor_diameter = 50.
        prob['turbineXw'] = turbineX
        prob['turbineYw'] = turbineY
        prob['rotorDiameter'] = np.array([rotor_diameter, rotor_diameter])
        prob['rotorDiameter'] = np.array([rotor_diameter, rotor_diameter])
        prob['model_params:exp_rate_multiplier'] = 1.0
        prob['model_params:wec_factor'] = 1.0

        prob.run_once()
        wspeed0 = prob['wtVelocity0'][1]
        prob['model_params:exp_rate_multiplier'] = 2.0
        prob.run_once()
        wspeed1 = prob['wtVelocity0'][1]

        self.assertEqual(wspeed1, wspeed0)

    def test_no_change_in_deficit_by_wake_diameter_multiplier_at_center(self):
        prob = self.prob
        turbineX = np.array([0., 400.])
        turbineY = np.array([0., 0.])
        rotor_diameter = 50.
        prob['turbineXw'] = turbineX
        prob['turbineYw'] = turbineY
        prob['rotorDiameter'] = np.array([rotor_diameter, rotor_diameter])
        prob['model_params:exp_rate_multiplier'] = 1.0
        prob['model_params:wec_factor'] = 1.0

        prob.run_once()
        wspeed0 = prob['wtVelocity0'][1]
        prob['model_params:exp_rate_multiplier'] = 2.0
        prob.run_once()
        wspeed1 = prob['wtVelocity0'][1]

        self.assertEqual(wspeed1, wspeed0)

    def test_increase_deficit_by_wake_diameter_expansion(self):
        prob = self.prob
        turbineX = np.array([0., 400.])
        turbineY = np.array([0., 100.])
        rotor_diameter = 50.
        prob['turbineXw'] = turbineX
        prob['turbineYw'] = turbineY
        prob['rotorDiameter'] = np.array([rotor_diameter, rotor_diameter])
        prob['model_params:exp_rate_multiplier'] = 1.0
        prob['model_params:wec_factor'] = 1.0

        prob.run_once()
        wspeed0 = prob['wtVelocity0'][1]
        prob['model_params:wec_factor'] = 2.0
        prob.run_once()
        wspeed1 = prob['wtVelocity0'][1]

        self.assertGreater(wspeed0, wspeed1)

    def test_increase_deficit_by_wake_expansion_rate_multiplier(self):
        prob = self.prob
        turbineX = np.array([0., 400.])
        turbineY = np.array([0., 100.])
        rotor_diameter = 50.
        prob['turbineXw'] = turbineX
        prob['turbineYw'] = turbineY
        prob['rotorDiameter'] = np.array([rotor_diameter, rotor_diameter])
        prob['model_params:exp_rate_multiplier'] = 1.0
        prob['model_params:wec_factor'] = 1.0

        prob.run_once()
        prob['model_params:wec_factor'] = 1.0
        wspeed0 = prob['wtVelocity0'][1]
        prob['model_params:exp_rate_multiplier'] = 2.0
        prob.run_once()
        wspeed1 = prob['wtVelocity0'][1]

        self.assertGreater(wspeed0, wspeed1)



if __name__ == "__main__":

    unittest.main(verbosity=2)