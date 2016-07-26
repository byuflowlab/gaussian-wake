import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from openmdao.api import Problem

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


if __name__ == "__main__":

    nTurbines = 2
    nDirections = 1

    rotorDiameter = 126.4
    rotorArea = np.pi*rotorDiameter*rotorDiameter/4.0
    axialInduction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axialInduction*(1.0-axialInduction)
    generator_efficiency = 0.944

    # Define turbine characteristics
    axialInduction = np.array([axialInduction, axialInduction])
    rotorDiameter = np.array([rotorDiameter, rotorDiameter])
    generatorEfficiency = np.array([generator_efficiency, generator_efficiency])
    yaw = np.array([0., 0.])

    # Define site measurements
    wind_direction = 270.-0.523599*180./np.pi
    wind_speed = 8.    # m/s
    air_density = 1.1716

    Ct = np.array([CT, CT])
    Cp = np.array([CP, CP])

    gauss_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=gauss_wrapper, wake_model_options={'nSamples': 0}, datasize=0,
                               params_IdepVar_func=add_gauss_params_IndepVarComps,
                               params_IndepVar_args={}))

    floris_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=floris_wrapper, wake_model_options=None, datasize=0,
                               params_IdepVar_func=add_floris_params_IndepVarComps,
                               params_IndepVar_args={}))

    probs = [gauss_prob, floris_prob]
    for prob in probs:
        prob.setup()

        turbineX = np.array([1118.1, 1881.9])
        turbineY = np.array([1279.5, 1720.5])

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['air_density'] = air_density
        prob['Cp_in'] = Cp
        prob['Ct_in'] = Ct
        prob['windSpeeds'] = np.array([wind_speed])
        prob['windDirections'] = np.array([wind_direction])

    # gauss_prob['model_params:ke'] = 0.052
    # gauss_prob['model_params:spread_angle'] = 6.
    # gauss_prob['model_params:rotation_offset_angle'] = 2.0

    # for axialInd calc only
    # gauss_prob['model_params:ke'] = 0.050688
    # gauss_prob['model_params:spread_angle'] = 7.562716
    # gauss_prob['model_params:rotation_offset_angle'] = 3.336568

    # for axialInd and inflow adjust
    # gauss_prob['model_params:ke'] = 0.052333
    # gauss_prob['model_params:spread_angle'] =  8.111330
    # gauss_prob['model_params:rotation_offset_angle'] = 2.770265

    # for inflow adjust only
    # gauss_prob['model_params:ke'] = 0.052230
    # gauss_prob['model_params:spread_angle'] =  6.368191
    # gauss_prob['model_params:rotation_offset_angle'] = 1.855112

    # for added n_st_dev param #1
    # gauss_prob['model_params:ke'] = 0.050755
    # gauss_prob['model_params:spread_angle'] = 11.205766#*0.97
    # gauss_prob['model_params:rotation_offset_angle'] = 3.651790
    # gauss_prob['model_params:n_std_dev'] = 9.304371

    # for added n_st_dev param #2
    # gauss_prob['model_params:ke'] = 0.051010
    # gauss_prob['model_params:spread_angle'] = 11.779591
    # gauss_prob['model_params:rotation_offset_angle'] = 3.564547
    # gauss_prob['model_params:n_std_dev'] = 9.575505

    # for decoupled ky with n_std_dev = 4
    # gauss_prob['model_params:ke'] = 0.051145
    # gauss_prob['model_params:spread_angle'] = 2.617982
    # gauss_prob['model_params:rotation_offset_angle'] = 3.616082
    # gauss_prob['model_params:ky'] = 0.211496

    # for integrating for decoupled ky with n_std_dev = 4, linear, integrating
    # gauss_prob['model_params:ke'] = 0.016969
    # gauss_prob['model_params:spread_angle'] = 0.655430
    # gauss_prob['model_params:rotation_offset_angle'] = 3.615754
    # gauss_prob['model_params:ky'] = 0.195392

    # for integrating for decoupled ky with n_std_dev = 4, linear, integrating
    # gauss_prob['model_params:ke'] = 0.008858
    # gauss_prob['model_params:spread_angle'] = 0.000000
    # gauss_prob['model_params:rotation_offset_angle'] = 4.035276
    # gauss_prob['model_params:ky'] = 0.199385

    # for decoupled ke with n_std_dev=4, linear, not integrating
    # gauss_prob['model_params:ke'] = 0.051190
    # gauss_prob['model_params:spread_angle'] = 2.619202
    # gauss_prob['model_params:rotation_offset_angle'] = 3.629337
    # gauss_prob['model_params:ky'] = 0.211567


    # for decoupled ky with n_std_dev = 4, error = 1332.49, not integrating, power law
    gauss_prob['model_params:ke'] = 0.051360
    gauss_prob['model_params:rotation_offset_angle'] = 3.197348
    gauss_prob['model_params:Dw0'] = 1.804024
    gauss_prob['model_params:m'] = 0.0

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
    gauss_prob['model_params:spread_mode'] = 'power'
    gauss_prob['model_params:n_std_dev'] = 4

    ICOWESdata = loadmat('../data/YawPosResults.mat')
    yawrange = ICOWESdata['yaw'][0]

    GaussianPower = list()
    FlorisPower = list()

    import time

    t1 = time.time()
    for i in range(0, 100):
        gauss_prob.run()
    t2 = time.time()
    for i in range(0, 100):
        floris_prob.run()
    t3 = time.time()
    # gauss time:  0.0580031871796
    # floris time:  0.10697388649

    print 'gauss time: ', t2-t1
    print 'floris time: ', t3-t2

    # quit()

    for yaw1 in yawrange:

        for prob in probs:
            prob['yaw0'] = np.array([yaw1, 0.0])
            prob.run()

        GaussianPower.append(list(gauss_prob['wtPower0']))
        FlorisPower.append(list(floris_prob['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    # print FlorisPower

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)
    power_scalar = 1E-3
    axes[0, 0].plot(yawrange.transpose(), FlorisPower[:, 0]*power_scalar, 'b', yawrange.transpose(), SOWFApower[:, 0]*power_scalar, 'o', mec='b', mfc='none')
    axes[0, 0].plot(yawrange.transpose(), FlorisPower[:, 1]*power_scalar, 'b', yawrange.transpose(), SOWFApower[:, 1]*power_scalar, '^', mec='b', mfc='none')
    axes[0, 0].plot(yawrange.transpose(), FlorisPower[:, 0]*power_scalar+FlorisPower[:, 1]*power_scalar, '-k', yawrange.transpose(), SOWFApower[:, 0]*power_scalar
                    + SOWFApower[:, 1]*power_scalar, 'ko')
    axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 0]*power_scalar, '--r')
    axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 1]*power_scalar, '--r')
    axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 0]*power_scalar+GaussianPower[:, 1]*power_scalar, '--k')
    axes[0, 0].set_xlabel('yaw angle (deg.)')
    axes[0, 0].set_ylabel('Power (MW)')
    # error_turbine2 = np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))
    posrange = ICOWESdata['pos'][0]

    for prob in probs:
        prob['yaw0'] = np.array([0.0, 0.0])

    GaussianPower = list()
    FlorisPower = list()

    for pos2 in posrange:
        # Define turbine locations and orientation
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1881.9])
        Yinit = np.array([1279.5, 1720.5])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))
        for prob in probs:
            prob['turbineX'] = XY[0, :]
            prob['turbineY'] = XY[1, :]
            prob.run()

        GaussianPower.append(list(gauss_prob['wtPower0']))
        FlorisPower.append(list(floris_prob['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    # print error_turbine2

    axes[0, 1].plot(posrange/rotorDiameter[0], FlorisPower[:, 0]*power_scalar, 'b', posrange/rotorDiameter[0], SOWFApower[:, 0]*power_scalar, 'o', mec='b', mfc='none')
    axes[0, 1].plot(posrange/rotorDiameter[0], FlorisPower[:, 1]*power_scalar, 'b', posrange/rotorDiameter[0], SOWFApower[:, 1]*power_scalar, '^', mec='b', mfc='none')
    axes[0, 1].plot(posrange/rotorDiameter[0], FlorisPower[:, 0]*power_scalar+FlorisPower[:, 1]*power_scalar, 'k-', posrange/rotorDiameter[0], SOWFApower[:, 0]*power_scalar+SOWFApower[:, 1]*power_scalar, 'ko')

    axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 0]*power_scalar, '--r')
    axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 1]*power_scalar, '--r')
    axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 0]*power_scalar+GaussianPower[:, 1]*power_scalar, '--k')

    axes[0, 1].set_xlabel('y/D')
    axes[0, 1].set_ylabel('Power (MW)')

    posrange = np.linspace(-3.*rotorDiameter[0], 3.*rotorDiameter[0], num=1000)

    for prob in probs:
        prob['yaw0'] = np.array([0.0, 0.0])
        prob['windDirections'] = np.array([270.])
        prob['turbineX'] = np.array([0, 7.*rotorDiameter[0]])

    GaussianVelocity = list()
    FlorisVelocity = list()

    for pos2 in posrange:
        for prob in probs:
            prob['turbineY'] = np.array([0, pos2])
            prob.run()

        GaussianVelocity.append(list(gauss_prob['wtVelocity0']))
        FlorisVelocity.append(list(floris_prob['wtVelocity0']))

    FlorisVelocity = np.array(FlorisVelocity)
    GaussianVelocity = np.array(GaussianVelocity)

    axes[1, 0].plot(posrange/rotorDiameter[0], FlorisVelocity[:, 1], 'b', label='Floris')
    axes[1, 0].plot(posrange/rotorDiameter[0], GaussianVelocity[:, 1], '--r', label='Gaussian')

    axes[1, 0].set_ylim([6.0, 8.5])
    axes[1, 0].set_xlim([-3.0, 3.0])

    axes[1, 0].set_xlabel('y/D')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    # plt.legend()
    # plt.show()

    posrange = np.linspace(-1.*rotorDiameter[0], 30.*rotorDiameter[0], num=2000)

    for prob in probs:
        prob['turbineY'] = np.array([0, 0])

    GaussianVelocity = list()
    FlorisVelocity = list()

    for pos2 in posrange:

        for prob in probs:
            prob['turbineX'] = np.array([0, pos2])
            prob.run()

        GaussianVelocity.append(list(gauss_prob['wtVelocity0']))
        FlorisVelocity.append(list(floris_prob['wtVelocity0']))

    FlorisVelocity = np.array(FlorisVelocity)
    GaussianVelocity = np.array(GaussianVelocity)

    axes[1, 1].plot(posrange/rotorDiameter[0], FlorisVelocity[:, 1], 'b', label='Floris')
    axes[1, 1].plot(posrange/rotorDiameter[0], GaussianVelocity[:, 1], '--r', label='Gaussian')
    axes[1, 1].plot(np.array([7.0, 7.0]), np.array([0.0, 9.0]), ':k', label='Tuning Point')

    plt.xlabel('x/D')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc=4)
    plt.show()