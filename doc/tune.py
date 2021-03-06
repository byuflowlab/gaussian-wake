import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from openmdao.api import Problem

from pyoptsparse import Optimization, SNOPT

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps


def tuning_obj_function(xdict={'ke': 0.052, 'spread_angle': 7.0, 'rotation_offset_angle': 2.0, 'ky': 6.0}, plot=False):

    global prob

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

    try:
        prob['model_params:ke'] = xdict['ke']
    except:
        temp = 0
    try:
        prob['model_params:spread_angle'] = xdict['spread_angle']
    except:
        temp = 0
    try:
        prob['model_params:rotation_offset_angle'] = xdict['rotation_offset_angle']
    except:
        temp = 0
    try:
        prob['model_params:n_std_dev'] = xdict['n_std_dev']
        # print prob['model_params:n_std_dev']
    except:
        # print "here here"
        # quit()
        temp = 0
    try:
        prob['model_params:ky'] = xdict['ky']
        # print prob['model_params:n_std_dev']
    except:
        # print "here here"
        prob['model_params:ky'] = xdict['ke']
    try:
        prob['model_params:Dw0'] = xdict['Dw0']
        # print prob['model_params:Dw0'], xdict['Dw0']
        # quit()
        # print prob['model_params:n_std_dev']
    except:
        # print "here here"
        # quit()
        temp = np.zeros(3)
    try:
        prob['model_params:m'] = xdict['m']
        # print prob['model_params:n_std_dev']
    except:
        # print "here here"
        # quit()
        temp = 0

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

    if plot:
        fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)
        axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 0], 'b', yawrange.transpose(), SOWFApower[:, 0], 'bo')
        axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 1], 'r', yawrange.transpose(), SOWFApower[:, 1], 'ro')
        axes[0, 0].plot(yawrange.transpose(), GaussianPower[:, 0]+GaussianPower[:, 1], 'k-', yawrange.transpose(), SOWFApower[:, 0]
                        + SOWFApower[:, 1], 'ko')
        axes[0, 0].set_xlabel('yaw angle (deg.)')
        axes[0, 0].set_ylabel('Power (kW)')
    # print yawrange
    # print GaussianPower[:, 0]
    # array = GaussianPower[:, 1]
    # print yawrange[2:-2]
    # error_turbine2 = 2.*np.sum(np.abs(GaussianPower[:, 1][2:-2] - SOWFApower[:, 1][2:-2]))
    weights = np.ones_like(GaussianPower[:, 1])
    # weights[-3:] *= 0.5
    # print weights
    # quit()
    error_turbine2 = np.sum(np.abs(GaussianPower[:, 1]*weights - SOWFApower[:, 1]))

    posrange = ICOWESdata['pos'][0]

    prob['yaw0'] = np.array([0.0, 0.0])

    GaussianPower = list()

    # print posrange.size, yawrange.size
    # quit()

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

    # print posrange[1:-2]
    # quit()
    weight = np.ones_like(GaussianPower[:, 0])
    # weight[-1] *= 0
    # weight[-2] *= 0
    # weight[1:3] *= 2.0
    # weight[-4:-2] *= 2.0
    error_turbine2 += np.sum(weight*np.abs(GaussianPower[:, 1] - SOWFApower[:, 1]))

    if plot:
        axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 0], 'b', posrange/rotorDiameter[0], SOWFApower[:, 0], 'bo')
        axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 1], 'r', posrange/rotorDiameter[0], SOWFApower[:, 1], 'ro')
        axes[0, 1].plot(posrange/rotorDiameter[0], GaussianPower[:, 0]+GaussianPower[:, 1], 'k-', posrange/rotorDiameter[0], SOWFApower[:, 0]+SOWFApower[:, 1], 'ko')

        axes[0, 1].set_xlabel('y/D')
        axes[0, 1].set_ylabel('Power (kW)')

        plt.show()

    print error_turbine2

    funcs = {'obj': error_turbine2}
    fail = False
    return funcs, fail


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

    global prob
    prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=gauss_wrapper, wake_model_options={'nSamples': 0}, datasize=0,
                               params_IdepVar_func=add_gauss_params_IndepVarComps,
                               params_IndepVar_args={}))

    prob.setup()
    prob['model_params:integrate'] = False
    prob['model_params:spread_mode'] = 'bastankhah'
    prob['model_params:yaw_mode'] = 'bastankhah'
    prob['model_params:n_std_dev'] = 4.
    # prob['model_params:m'] = 0.33
    # prob['model_params:Dw0'] = 1.3

    tuning_obj_function(plot=True)

    # initialize optimization problem
    optProb = Optimization('Tuning Gaussian Model to SOWFA', tuning_obj_function)
    optProb.addVarGroup('ke', 1, lower=0.0, upper=1.0, value=0.152, scalar=1)
    # optProb.addVarGroup('spread_angle', 1, lower=0.0, upper=30.0, value=3.0, scalar=1)
    optProb.addVarGroup('rotation_offset_angle', 1, lower=0.0, upper=5.0, value=1.5, scalar=1)
    # optProb.addVarGroup('ky', 1, lower=0.0, upper=20.0, value=0.1, scalar=1E-4)
    # optProb.addVarGroup('Dw0', 3, lower=np.zeros(3), upper=np.ones(3)*20., value=np.array([1.3, 1.3, 1.3]))#, scalar=1E-2)
    # optProb.addVarGroup('m', 1, lower=0.1, upper=20.0, value=0.33, scalar=1E-3)

    # add objective
    optProb.addObj('obj', scale=1E-3)

    # initialize optimizer
    snopt = SNOPT()

    # run optimizer
    sol = snopt(optProb, sens='FD')

    print sol

    # tuning_obj_function(xdict={'ke': sol.xStar['ke'], 'spread_angle': sol.xStar['spread_angle'],
    #                            'rotation_offset_angle': sol.xStar['rotation_offset_angle'],
    #                            'ky': sol.xStar['ky']}, plot=True)

    tuning_obj_function(xdict=sol.xStar, plot=True)