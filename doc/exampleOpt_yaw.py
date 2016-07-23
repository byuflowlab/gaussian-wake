import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from openmdao.api import Problem, pyOptSparseDriver

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps
from wakeexchange.jensen import jensen_wrapper, add_jensen_params_IndepVarComps


if __name__ == "__main__":

    turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])
    nTurbines = turbineX.size

    rotor_diameter = 126.4
    axial_induction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axial_induction*(1.0-axial_induction)
    generator_efficiency = 0.944
    yaw_init = 0.0

    # Define turbine characteristics
    axialInduction = np.zeros(nTurbines) + axial_induction
    rotorDiameter = np.zeros(nTurbines) + rotor_diameter
    generatorEfficiency = np.zeros(nTurbines) + generator_efficiency
    yaw = np.zeros(nTurbines) + yaw_init
    Ct = np.zeros(nTurbines) + CT
    Cp = np.zeros(nTurbines) + CP

    # Define site measurements
    nDirections = 1
    wind_direction = 270.-0.523599*180./np.pi
    wind_speed = 8.    # m/s
    air_density = 1.1716    

    # initialize problems
    gauss_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=gauss_wrapper, wake_model_options={'nSamples': 0}, datasize=0,
                               params_IdepVar_func=add_gauss_params_IndepVarComps, force_fd=True,
                               params_IndepVar_args={}))

    floris_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=floris_wrapper, wake_model_options=None, datasize=0,
                               params_IdepVar_func=add_floris_params_IndepVarComps,
                               params_IndepVar_args={}))

    jensen_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=jensen_wrapper, wake_model_options={'variant': 'CosineYaw_1R',
                                                                              'radius multiplier': 1.0}, datasize=0,
                               params_IdepVar_func=add_jensen_params_IndepVarComps,
                               params_IndepVar_args={'use_angle': True}))

    probs = [gauss_prob, floris_prob, jensen_prob]
    names = ['gauss', 'floris', 'jensen']

    for indx, prob in enumerate(probs):

        # set up optimizer
        prob.driver = pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'SNOPT'
        prob.driver.add_objective('obj', scaler=1E-5)
    
        # set optimizer options
        prob.driver.opt_settings['Verify level'] = 3
        prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptYaw_%s.out' % names[indx]
        prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptYaw_%s.out' % names[indx]
        prob.driver.opt_settings['Major iterations limit'] = 1000
    
        # select design variables
        for direction_id in range(0, nDirections):
            prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1)

        prob.setup()

        if names[indx] is 'gauss':
            # gauss_prob['model_params:ke'] = 0.052
            # gauss_prob['model_params:spread_angle'] = 6.
            # gauss_prob['model_params:rotation_offset_angle'] = 2.0

            # gauss_prob['model_params:ke'] = 0.050755
            # gauss_prob['model_params:spread_angle'] = 11.205766
            # gauss_prob['model_params:rotation_offset_angle'] = 3.651790
            # gauss_prob['model_params:n_std_dev'] = 9.304371

            # gauss_prob['model_params:ke'] = 0.051010
            # gauss_prob['model_params:spread_angle'] = 11.779591
            # gauss_prob['model_params:rotation_offset_angle'] = 3.564547
            # gauss_prob['model_params:n_std_dev'] = 9.575505

            # using ky with n_std_dev = 6
            # gauss_prob['model_params:ke'] = 0.051115
            # gauss_prob['model_params:spread_angle'] = 5.967284
            # gauss_prob['model_params:rotation_offset_angle'] = 3.597926
            # gauss_prob['model_params:ky'] = 0.494776

            # using ky with n_std_dev = 4
            # gauss_prob['model_params:ke'] = 0.051030
            # gauss_prob['model_params:spread_angle'] = 2.584067
            # gauss_prob['model_params:rotation_offset_angle'] = 3.618665
            # gauss_prob['model_params:ky'] = 0.214723

            # using ky with n_std_dev = 3
            # gauss_prob['model_params:ke'] = 0.051079
            # gauss_prob['model_params:spread_angle'] = 0.943942
            # gauss_prob['model_params:rotation_offset_angle'] = 3.579857
            # gauss_prob['model_params:ky'] = 0.078069

            # for decoupled ky with n_std_dev = 4
            # gauss_prob['model_params:ke'] = 0.051145
            # gauss_prob['model_params:spread_angle'] = 2.617982
            # gauss_prob['model_params:rotation_offset_angle'] = 3.616082
            # gauss_prob['model_params:ky'] = 0.211496

            # for decoupled ky with n_std_dev = 6 and double diameter wake at rotor pos
            # gauss_prob['model_params:ke'] = 0.051030
            # gauss_prob['model_params:spread_angle'] = 1.864696
            # gauss_prob['model_params:rotation_offset_angle'] = 3.362729
            # gauss_prob['model_params:ky'] = 0.193011

            # for integrating for decoupled ky with n_std_dev = 4, error = 1034.3
            # gauss_prob['model_params:ke'] = 0.007523
            # gauss_prob['model_params:spread_angle'] = 1.876522
            # gauss_prob['model_params:rotation_offset_angle'] = 3.633083
            # gauss_prob['model_params:ky'] = 0.193160

            # for integrating using power law
            # gauss_prob['model_params:ke'] = 0.033165
            # gauss_prob['model_params:rotation_offset_angle'] = 3.328051
            # gauss_prob['model_params:Dw0'] = 1.708328
            # gauss_prob['model_params:m'] = 0.0

            # for decoupled ke with n_std_dev=4, linear, not integrating
            # gauss_prob['model_params:ke'] = 0.051190
            # gauss_prob['model_params:spread_angle'] = 2.619202
            # gauss_prob['model_params:rotation_offset_angle'] = 3.629337
            # gauss_prob['model_params:ky'] = 0.211567

            # for integrating for decoupled ky with n_std_dev = 4, error = 1034.3, linear, integrating
            # gauss_prob['model_params:ke'] = 0.008858
            # gauss_prob['model_params:spread_angle'] = 0.000000
            # gauss_prob['model_params:rotation_offset_angle'] = 4.035276
            # gauss_prob['model_params:ky'] = 0.199385

            # for decoupled ky with n_std_dev = 4, error = 1332.49, not integrating, power law
            # gauss_prob['model_params:ke'] = 0.051360
            # gauss_prob['model_params:rotation_offset_angle'] = 3.197348
            # gauss_prob['model_params:Dw0'] = 1.804024
            # gauss_prob['model_params:m'] = 0.0

            # for decoupled ky with n_std_dev = 4, error = 1630.8, with integrating, power law
            # gauss_prob['model_params:ke'] = 0.033165
            # gauss_prob['model_params:rotation_offset_angle'] = 3.328051
            # gauss_prob['model_params:Dw0'] = 1.708328
            # gauss_prob['model_params:m'] = 0.0

            # for decoupled ky with n_std_dev = 4, error = 1140.59, not integrating, power law for expansion,
            # linear for yaw
            # gauss_prob['model_params:ke'] = 0.050741
            # gauss_prob['model_params:rotation_offset_angle'] = 3.628737
            # gauss_prob['model_params:Dw0'] = 0.846582
            # gauss_prob['model_params:ky'] = 0.207734

            # for decoupled ky with n_std_dev = 4, error = 1058.73, integrating, power law for expansion,
            # linear for yaw
            # gauss_prob['model_params:ke'] = 0.016129
            # gauss_prob['model_params:rotation_offset_angle'] = 3.644356
            # gauss_prob['model_params:Dw0'] = 0.602132
            # gauss_prob['model_params:ky'] = 0.191178

            gauss_prob['model_params:integrate'] = False
            gauss_prob['model_params:spread_mode'] = 'linear'
            gauss_prob['model_params:n_std_dev'] = 4



        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['Cp_in'] = Cp
        prob['Ct_in'] = Ct
        prob['windSpeeds'] = np.array([wind_speed])
        prob['windDirections'] = np.array([wind_direction])

        # prob.run_once()
        prob.run()

    for indx, prob in enumerate(probs):
        print names[indx], prob['yaw0']
        print 'power', np.sum(prob['wtPower0'])

