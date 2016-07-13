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

    gauss_prob['model_params:ke'] = 0.052
    gauss_prob['model_params:spread_angle'] = 6.
    gauss_prob['model_params:rotation_offset_angle'] = 2.0

    ICOWESdata = loadmat('../data/YawPosResults.mat')
    yawrange = ICOWESdata['yaw'][0]

    GaussianPower = list()
    FlorisPower = list()

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