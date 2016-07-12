import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from openmdao.api import Problem

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps


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

    prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=gauss_wrapper, wake_model_options={'nSamples': 0}, datasize=0,
                               params_IdepVar_func=add_gauss_params_IndepVarComps,
                               params_IndepVar_args={}))

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

    prob['model_params:ke'] = 0.052
    prob['model_params:spread_angle'] = 6.
    prob['model_params:rotation_offset_angle'] = 2.0

    ICOWESdata = loadmat('../data/YawPosResults.mat')
    yawrange = ICOWESdata['yaw'][0]

    GaussianPower = list()

    for yaw1 in yawrange:

        prob['yaw0'] = np.array([yaw1, 0.0])
        prob.run()
        wt_power = prob['wtPower0']

        GaussianPower.append(list(wt_power))

    GaussianPower = np.array(GaussianPower)

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)
    axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 0], 'b', yawrange.transpose(), SOWFApower[:, 0], 'bo')
    axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 1], 'r', yawrange.transpose(), SOWFApower[:, 1], 'ro')
    axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 0]+GaussianPower[:, 1], 'k-', yawrange.transpose(), SOWFApower[:, 0]
                    + SOWFApower[:, 1], 'ko')
    axes[0, 0].set_xlabel('yaw angle (deg.)')
    axes[0, 0].set_ylabel('Power (kW)')
    # error_turbine2 = np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))
    posrange = ICOWESdata['pos'][0]

    prob['yaw0'] = np.array([0.0, 0.0])

    GaussianPower = list()

    for pos2 in posrange:
        # Define turbine locations and orientation
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1881.9])
        Yinit = np.array([1279.5, 1720.5])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))

        prob['turbineX'] = XY[0, :]
        prob['turbineY'] = XY[1, :]

        prob.run()

        wt_power = prob['wtPower0']

        GaussianPower.append(list(wt_power))

    GaussianPower = np.array(GaussianPower)

    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    # print error_turbine2

    axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 0], 'b', posrange/rotorDiameter[0], SOWFApower[:, 0], 'bo')
    axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 1], 'r', posrange/rotorDiameter[0], SOWFApower[:, 1], 'ro')
    axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 0]+GaussianPower[:, 1], 'k-', posrange/rotorDiameter[0], SOWFApower[:, 0]+SOWFApower[:, 1], 'ko')

    axes[0, 1].set_xlabel('y/D')
    axes[0, 1].set_ylabel('Power (kW)')
    #
    # posrange = np.linspace(-3.*rotorDiameter[0], 3.*rotorDiameter[0], num=1000)
    # yaw = np.array([0.0, 0.0])
    # wind_direction = 0.0
    # FLORISvelocity = list()
    # # FLORISvelocityStat = list()
    # FLORISvelocityLES = list()
    # for pos2 in posrange:
    #
    #     turbineX = np.array([0, 7.*rotorDiameter[0]])
    #     turbineY = np.array([0, pos2])
    #     # print turbineX, turbineY
    #
    #     # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
    #     #                                                                                 air_density, rotorDiameter,
    #     #                                                                                 yaw, Cp, axialInduction,
    #     #                                                                                 turbineX, turbineY)
    #
    #     # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
    #     #                                                                                 air_density, rotorDiameter,
    #     #                                                                                 yaw, Cp, axialInduction,
    #     #                                                                                 turbineX, turbineY)
    #
    #     velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #
    #
    #
    #     velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     # print 'power = ', myFloris.root.dir0.unknowns['wt_power']
    #
    #     FLORISvelocity.append(list(velocitiesTurbines))
    #     # FLORISvelocityStat.append(list(velocitiesTurbines_stat))
    #     # FLORISvelocityLES.append(list(velocitiesTurbines_LES))
    #     FLORISvelocityGAZ.append(list(velocitiesTurbines_GAZ))
    #     FLORISvelocityGAR.append(list(velocitiesTurbines_GAR))
    #     FLORISvelocityGARZ.append(list(velocitiesTurbines_GARZ))
    #     FLORISvelocityGH.append(list(velocitiesTurbines_GH))
    #     FLORISvelocityCos.append(list(velocitiesTurbines_Cos))
    #
    #
    # FLORISvelocity = np.array(FLORISvelocity)
    # # FLORISvelocityStat = np.array(FLORISvelocityStat)
    # # FLORISvelocityLES = np.array(FLORISvelocityLES)
    # FLORISvelocityGAZ = np.array(FLORISvelocityGAZ)
    # FLORISvelocityGAR = np.array(FLORISvelocityGAR)
    # FLORISvelocityGARZ = np.array(FLORISvelocityGARZ)
    # FLORISvelocityGH = np.array(FLORISvelocityGH)
    # FLORISvelocityCos = np.array(FLORISvelocityCos)
    #
    #
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocity[:, 1], 'b', label='Floris Original')
    # # axes[2].plot(posrange, FLORISvelocityStat[:, 1], 'g', label='added gaussian')
    # # axes[2].plot(posrange/rotorDiameter[0], FLORISvelocityLES[:, 1], 'c', label='Linear Exact Sector')
    # # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGAZ[:, 1], 'm', label='GAZ')
    # # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGAR[:, 1], 'c', label='GAR')
    # # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGARZ[:, 1], 'g', label='GARZ')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGH[:, 1], 'y', label='GH')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityCos[:, 1], 'r', label='Cos')
    #
    # axes[1, 0].set_xlabel('y/D')
    # axes[1, 0].set_ylabel('Velocity (m/s)')
    # # plt.legend()
    # # plt.show()
    #
    # posrange = np.linspace(-1.*rotorDiameter[0], 30.*rotorDiameter[0], num=2000)
    # yaw = np.array([0.0, 0.0])
    # wind_direction = 0.0
    # FLORISvelocity = list()
    # # FLORISvelocityStat = list()
    # FLORISvelocity = list()
    # # FLORISvelocityStat = list()
    # # FLORISvelocityLES = list()
    # FLORISvelocityGAZ = list()
    # FLORISvelocityGAR = list()
    # FLORISvelocityGARZ = list()
    # FLORISvelocityGH = list()
    # FLORISvelocityCos = list()
    # for pos2 in posrange:
    #
    #     turbineX = np.array([0, pos2])
    #     turbineY = np.array([0, 0])
    #     # print turbineX, turbineY
    #
    #     # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
    #     #                                                                                 air_density, rotorDiameter,
    #     #                                                                                 yaw, Cp, axialInduction,
    #     #                                                                                 turbineX, turbineY)
    #
    #     # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
    #     #                                                                                 air_density, rotorDiameter,
    #     #                                                                                 yaw, Cp, axialInduction,
    #     #                                                                                 turbineX, turbineY)
    #
    #     velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #
    #
    #
    #
    #     velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     # print 'power = ', myFloris.root.dir0.unknowns['wt_power']
    #
    #     FLORISvelocity.append(list(velocitiesTurbines))
    #     # FLORISvelocityStat.append(list(velocitiesTurbines_stat))
    #     # FLORISvelocityLES.append(list(velocitiesTurbines_LES))
    #     FLORISvelocityGAZ.append(list(velocitiesTurbines_GAZ))
    #     FLORISvelocityGAR.append(list(velocitiesTurbines_GAR))
    #     FLORISvelocityGARZ.append(list(velocitiesTurbines_GARZ))
    #     FLORISvelocityGH.append(list(velocitiesTurbines_GH))
    #     FLORISvelocityCos.append(list(velocitiesTurbines_Cos))
    #
    #
    # FLORISvelocity = np.array(FLORISvelocity)
    # # FLORISvelocityStat = np.array(FLORISvelocityStat)
    # # FLORISvelocityLES = np.array(FLORISvelocityLES)
    # FLORISvelocityGAZ = np.array(FLORISvelocityGAZ)
    # FLORISvelocityGAR = np.array(FLORISvelocityGAR)
    # FLORISvelocityGARZ = np.array(FLORISvelocityGARZ)
    # FLORISvelocityGH = np.array(FLORISvelocityGH)
    # FLORISvelocityCos = np.array(FLORISvelocityCos)
    #
    #
    # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocity[:, 1], 'b', label='Floris Original')
    # # axes[2].plot(posrange, FLORISvelocityStat[:, 1], 'g', label='added gaussian')
    # # axes[2].plot(posrange/rotorDiameter[0], FLORISvelocityLES[:, 1], 'c', label='Linear Exact Sector')
    # # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGAZ[:, 1], 'm', label='GAZ')
    # # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGAR[:, 1], 'c', label='GAR')
    # # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGARZ[:, 1], 'g', label='GARZ')
    # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGH[:, 1], 'y', label='GH')
    # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityCos[:, 1], 'r', label='Cos')
    # axes[1, 1].plot(np.array([7, 7]), np.array([2, 8]), '--k', label='tuning point')
    # print min(FLORISvelocity[:, 1])
    # plt.xlabel('x/D')
    # plt.ylabel('Velocity (m/s)')
    # plt.legend(loc=4)
    plt.show()