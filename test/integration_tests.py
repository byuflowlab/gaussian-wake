"""
Created by Jared J. Thomas, Sep. 2018.
FLOW Lab
Brigham Young University
"""

import unittest
import numpy as np
from scipy.interpolate import UnivariateSpline

from plantenergy.OptimizationGroups import AEPGroup

from openmdao.api import Problem


class test_guass(unittest.TestCase):

    def setUp(self):
        try:
            from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
            self.working_import = True
        except:
            self.working_import = False

        # define turbine locations in global reference frame
        # this is the 5 deg rotated from free stream wind farm from Gebraad 2014 CFD study
        # wind to 30 deg from east, farm rotated to 35 deg from east
        turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
        turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

        hubHeight = np.zeros_like(turbineX)+90.
        # import matplotlib.pyplot as plt
        # plt.plot(turbineX, turbineY, 'o')
        # print np.arctan((turbineY[5]-turbineY[0])/(turbineX[5]-turbineX[0]))*180./np.pi
        # print 0.523599*180./np.pi
        # plt.show()
        # quit()
        # plt.plot(turbineX, turbineY, 'o')
        # plt.plot(np.array([0.0, ]))
        # plt.show()

        # initialize input variable arrays
        nTurbines = turbineX.size
        rotorDiameter = np.zeros(nTurbines)
        axialInduction = np.zeros(nTurbines)
        Ct = np.zeros(nTurbines)
        Cp = np.zeros(nTurbines)
        generatorEfficiency = np.zeros(nTurbines)
        yaw = np.zeros(nTurbines)

        # define initial values
        for turbI in range(0, nTurbines):
            rotorDiameter[turbI] = 126.4            # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 1.#0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.0                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem
        nRotorPoints = 100

        # define turbine size
        rotor_diameter = 126.4  # (m)
        hub_height = 90.0

        z_ref = 80.0  # m
        z_0 = 0.0

        # load performance characteristics
        cut_in_speed = 3.  # m/s
        rated_power = 5000.  # kW
        generator_efficiency = 0.944
        input_directory = "./input_files/"
        filename = input_directory + "NREL5MWCPCT_dict.p"
        # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
        import cPickle as pickle

        data = pickle.load(open(filename, "rb"))
        ct_curve = np.zeros([data['wind_speed'].size, 2])
        ct_curve_wind_speed = data['wind_speed']
        ct_curve_ct = data['CT']

        # cp_curve_cp = data['CP']
        # cp_curve_wind_speed = data['wind_speed']

        loc0 = np.where(data['wind_speed'] < 11.55)
        loc1 = np.where(data['wind_speed'] > 11.7)

        cp_curve_cp = np.hstack([data['CP'][loc0], data['CP'][loc1]])
        cp_curve_wind_speed = np.hstack([data['wind_speed'][loc0], data['wind_speed'][loc1]])
        cp_curve_spline = UnivariateSpline(cp_curve_wind_speed, cp_curve_cp, ext='const')
        cp_curve_spline.set_smoothing_factor(.000001)

        wake_model_options = {'nSamples': 0,
                              'nRotorPoints': nRotorPoints,
                              'use_ct_curve': True,
                              'ct_curve_ct': ct_curve_ct,
                              'ct_curve_wind_speed': ct_curve_wind_speed,
                              'interp_type': 1,
                              'use_rotor_components': False,
                              'differentiable': True,
                              'verbose': False}

        prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=gauss_wrapper,
                                     wake_model_options=wake_model_options, datasize=0, use_rotor_components=False,
                                     params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
                                     params_IndepVar_args={'nRotorPoints': nRotorPoints}))

        # initialize problem
        prob.setup(check=True)

        if nRotorPoints > 1:
            from plantenergy.utilities import sunflower_points
            prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['hubHeight'] = hubHeight
        prob['yaw0'] = yaw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = np.array([wind_speed])
        prob['model_params:I'] = 0.06
        prob['model_params:z_ref'] = 90.
        prob['model_params:z_0'] = 0.001
        prob['model_params:wake_model_version'] = 2016.
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['cp_curve_cp'] = cp_curve_cp
        prob['cp_curve_wind_speed'] = cp_curve_wind_speed
        cutInSpeeds = np.ones(nTurbines) * cut_in_speed
        prob['cut_in_speed'] = cutInSpeeds
        ratedPowers = np.ones(nTurbines) * rated_power
        prob['rated_power'] = ratedPowers

        prob['model_params:wake_combination_method'] = 1
        prob['model_params:ti_calculation_method'] = 4
        prob['model_params:wake_model_version'] = 2016
        prob['model_params:wec_factor'] = 1.0
        prob['model_params:calc_k_star'] = True
        prob['model_params:sort'] = True
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = 0.022
        prob['model_params:kz'] = 0.022
        prob['model_params:print_ti'] = False
        prob['model_params:shear_exp'] = 0.15
        prob['model_params:I'] = 0.06
        prob['model_params:sm_smoothing'] = 700
        if nRotorPoints > 1:
            prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)

        prob['model_params:exp_rate_multiplier'] = 0.0

        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp

        # run the problem
        prob.run()

        self.prob = prob

    def testImport(self):
        self.assertEqual(self.working_import, True, "gauss_wrapper Import Failed")

    def testRun(self):
        # data from Gebraad et al 2014
        # 0.9954426957301689, 1.8739635157545602
        # 2.0018487329887518, 2.0398009950248754
        # 2.9960347419839835, 1.2305140961857381
        # 3.9981577198314526, 1.1575456053067994
        # 4.99994999064341, 1.2205638474295195
        # 5.992119493324558, 1.240464344941957
        np.testing.assert_allclose(self.prob['wtPower0']*1E-3, np.array([1.87, 2.03, 1.23, 1.15, 1.22, 1.24]), rtol=1E-0, atol=1E-0)


if __name__ == "__main__":
    unittest.main(verbosity=2)