from openmdao.api import Problem
from wakeexchange.GeneralWindFarmGroups import AEPGroup
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps

import time
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([0., 1000.])
    turbineY = np.array([0., 0.])

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = 126.4            # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        # Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        Cp[turbI] = 0.7737 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 1.0#0.944
        yaw[turbI] = 0.     # deg.

    # Define flow properties
    wind_speed = 8.0        # m/s
    air_density = 1.1716    # kg/m^3
    # wind_direction = 240    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    wind_direction = 270.    # deg (N = 0 deg., using direction FROM, as in met-mast data)
    print wind_direction
    wind_frequency = 1.    # probability of wind in this direction at this speed

    # set up problem
    gauss_prob = Problem(root=AEPGroup(nTurbs, nDirections=1, use_rotor_components=False, datasize=0,
                 differentiable=False, optimizingLayout=False, nSamples=0, wake_model=gauss_wrapper,
                 wake_model_options=None, params_IdepVar_func=add_gauss_params_IndepVarComps,
                 params_IndepVar_args=None))

    floris_prob = Problem(root=AEPGroup(nTurbines=nTurbs, nDirections=1, use_rotor_components=False,
                                      wake_model=floris_wrapper, wake_model_options=None, datasize=0,
                                      params_IdepVar_func=add_floris_params_IndepVarComps,
                                      params_IndepVar_args={}))

    # initialize problem
    gauss_prob.setup()
    floris_prob.setup()

    # assign values to turbine states
    gauss_prob['turbineX'] = turbineX
    gauss_prob['turbineY'] = turbineY
    gauss_prob['yaw0'] = yaw

    floris_prob['turbineX'] = turbineX
    floris_prob['turbineY'] = turbineY
    floris_prob['yaw0'] = yaw

    # assign values to constant inputs (not design variables)
    gauss_prob['rotorDiameter'] = rotorDiameter
    gauss_prob['axialInduction'] = axialInduction
    gauss_prob['generatorEfficiency'] = generatorEfficiency
    gauss_prob['windSpeeds'] = np.array([wind_speed])
    gauss_prob['air_density'] = air_density
    gauss_prob['windDirections'] = np.array([wind_direction])
    gauss_prob['windFrequencies'] = np.array([wind_frequency])
    gauss_prob['Ct_in'] = Ct
    gauss_prob['Cp_in'] = Cp

    floris_prob['rotorDiameter'] = rotorDiameter
    floris_prob['axialInduction'] = axialInduction
    floris_prob['generatorEfficiency'] = generatorEfficiency
    floris_prob['windSpeeds'] = np.array([wind_speed])
    floris_prob['air_density'] = air_density
    floris_prob['windDirections'] = np.array([wind_direction])
    floris_prob['windFrequencies'] = np.array([wind_frequency])
    floris_prob['Ct_in'] = Ct
    floris_prob['Cp_in'] = Cp
    # prob['model_params:spread_mode'] = 'bastankhah'
    # prob['model_params:yaw_mode'] = 'bastankhah'
    # gauss_prob['model_params:ky'] = 0.022 #0.7
    # gauss_prob['model_params:kz'] = 0.022 #e0.7
    # gauss_prob['model_params:alpha'] = 2.32
    # gauss_prob['model_params:beta'] = 0.154
    # gauss_prob['model_params:I'] = 0.075 #0.1

    # run the problem
    print 'start Bastankhah run'
    tic = time.time()
    # gauss_prob.run()
    # floris_prob.run()
    toc = time.time()

    directions = np.arange(0.,360., 5.)
    positions = np.arange(0.,11.*rotorDiameter[0], 5.)

    gauss_dv = np.zeros([np.size(positions), 2])

    floris_dv = np.zeros([np.size(positions), 2])

    i = 0

    for p in positions:
        gauss_prob['turbineX'] = np.array([0., p])
        gauss_prob.run()
        gauss_dv[i] = (wind_speed - gauss_prob['wtVelocity0'])/wind_speed

        floris_prob['turbineX'] = np.array([0., p])
        floris_prob.run()
        floris_dv[i] = (wind_speed - floris_prob['wtVelocity0'])/wind_speed
        i += 1

    # for p in positions:
    #     gauss_prob['turbineY'] = np.array([p, 0.])
    #     gauss_prob.run()
    #     gauss_AEP[i] = gauss_prob['AEP']
    #     gauss_P1[i] = gauss_prob['wtPower0'][0]
    #     gauss_P2[i] = gauss_prob['wtPower0'][1]
    #
    #     floris_prob['turbineY'] = np.array([d, 0.])
    #     floris_prob.run()
    #     floris_AEP[i] = floris_prob['AEP']
    #     floris_P1[i] = floris_prob['wtPower0'][0]
    #     floris_P2[i] = floris_prob['wtPower0'][1]
    #     i += 1

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(positions/rotorDiameter[0], gauss_dv[:, 1])
    ax1.plot(positions/rotorDiameter[0], floris_dv[:, 1])
    ax1.legend(labels=['gAEP', 'fAEP'],loc='lower center')

    plt.plot()

    plt.get_current_fig_manager().show()
    plt.show()
