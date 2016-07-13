import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from openmdao.api import Problem, pyOptSparseDriver

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


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
    yaw_init = 1.0

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

    probs = [gauss_prob, floris_prob]
    names = ['gauss', 'floris']

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
            prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1E2)

        prob.setup()

        if names[indx] is 'gauss':
            gauss_prob['model_params:ke'] = 0.052
            gauss_prob['model_params:spread_angle'] = 6.
            gauss_prob['model_params:rotation_offset_angle'] = 2.0

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

        prob.run()

    for indx, prob in enumerate(probs):
        print names[indx], prob['yaw0']

