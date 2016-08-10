import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cPickle as pickle

from openmdao.api import Problem

from pyoptsparse import Optimization, SNOPT

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


def tuning_obj_function(xdict={'ke': 0.052, 'spread_angle': 7.0, 'rotation_offset_angle': 2.0, 'ky': 6.0}, plot=False):

    global prob
    global model

    set_param_vals(xdict)

    turbineX = np.array([1118.1, 1881.9])
    turbineY = np.array([1279.5, 1720.5])

    yaw_weight = 1.

     # load data
    ICOWESdata = loadmat('../data/YawPosResults.mat')
    with open('../data/yawPower.p', 'rb') as handle:
        yawrange_4D, SOWFApower_yaw_4D,  _, _ = pickle.load(handle)
    with open('../data/offset4DPower.p', 'rb') as handle:
        posrange_cs_4D, SOWFApower_cs_4D = pickle.load(handle)
    with open('../data/offset6DPower.p', 'rb') as handle:
        posrange_cs_6D, SOWFApower_cs_6D = pickle.load(handle)
    with open('../data/spacePower.p', 'rb') as handle:
        posrange_ds, SOWFApower_ds = pickle.load(handle)

    # set tuning params
    ICOWESvelocity = 8.0
    PFvelocity = 8.48673684
    PFvelocity = 8.38673684
    rotor_diameter = rotorDiameter[0]

    error_turbine2 = 0.0


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

    # ################## compare yaw ######################
    # 4D yaw
    yawrange = np.array(list(yawrange_4D))
    Power = list()

    # set to 4D positions and inflow velocity
    prob['turbineX'] = np.array([1118.1, 1556.0])
    prob['turbineY'] = np.array([1279.5, 1532.3])
    prob['windSpeeds'] = np.array([PFvelocity])

    for yaw1 in yawrange:
        prob['yaw0'] = np.array([yaw1, 0.0])
        prob.run()

        Power.append(list(prob['wtPower0']))

    Power = np.array(Power)

    # print FlorisPower

    SOWFApower = SOWFApower_yaw_4D*1E-3

    error_turbine2 += yaw_weight*np.sum((SOWFApower[:, 1]-Power[:, 1])**2)

    # 7D yaw
    yawrange = ICOWESdata['yaw'][0]
    # yawrange = ICOWESdata['yaw'][0, 2:-2]
    Power = list()

    # set to 7D positions
    prob['turbineX'] = np.array([1118.1, 1881.9])
    prob['turbineY'] = np.array([1279.5, 1720.5])
    prob['windSpeeds'] = np.array([ICOWESvelocity])

    # run analysis
    for yaw1 in yawrange:

        prob['yaw0'] = np.array([yaw1, 0.0])
        prob.run()

        Power.append(list(prob['wtPower0']))

    Power = np.array(Power)

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    error_turbine2 += yaw_weight*np.sum((SOWFApower[:, 1]-Power[:, 1])**2)
    # error_turbine2 += yaw_weight*np.sum((SOWFApower[2:-2, 1]-Power[:, 1])**2)

    # ################## compare position ######################
    PosPowFig, PosPowAx = plt.subplots(ncols=2, nrows=2, sharey=False)

    prob['yaw0'] = np.array([0.0, 0.0])
    prob['windSpeeds'] = np.array([PFvelocity])

    # position crosswind 4D
    posrange = np.array(list(posrange_cs_4D))
    Power = list()

    for pos2 in posrange:
        # Define turbine locations and orientation (4D)
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1556.0])
        Yinit = np.array([1279.5, 1532.3])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))

        prob['turbineX'] = XY[0, :]
        prob['turbineY'] = XY[1, :]
        prob.run()

        Power.append(list(prob['wtPower0']))


    Power = np.array(Power)

    SOWFApower = SOWFApower_cs_4D*1E-3

    error_turbine2 += np.sum((SOWFApower[:, 1]-Power[:, 1])**2)

    # position crosswind 6D
    posrange = np.array(list(posrange_cs_6D))

    Power = list()

    prob['windSpeeds'] = np.array([PFvelocity])

    for pos2 in posrange:
        # Define turbine locations and orientation (4D)
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1556.0])
        Yinit = np.array([1279.5, 1532.3])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))
        prob['turbineX'] = XY[0, :]
        prob['turbineY'] = XY[1, :]
        prob.run()

        Power.append(list(prob['wtPower0']))

    Power = np.array(Power)

    SOWFApower = SOWFApower_cs_6D*1E-3

    error_turbine2 += np.sum((SOWFApower[:, 1]-Power[:, 1])**2)

    # position crosswind 7D
    posrange = ICOWESdata['pos'][0]
    Power = list()

    prob['windSpeeds'] = np.array([ICOWESvelocity])

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

        Power.append(list(prob['wtPower0']))

    Power = np.array(Power)

    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    error_turbine2 += np.sum((SOWFApower[:, 1]-Power[:, 1])**2)

    # position downstream
    posrange = np.array(list(posrange_ds))*rotor_diameter

    Power = list()

    prob['windSpeeds'] = np.array([PFvelocity])
    prob['turbineY'] = np.array([0.0, 0.0])
    prob['windDirections'] = np.array([270.0])

    for pos2 in posrange:

        prob['turbineX'] = np.array([0.0, pos2])
        prob.run()

        Power.append(list(prob['wtPower0']))

    Power = np.array(Power)

    SOWFApower = SOWFApower_ds*1E-3

    error_turbine2 += np.sum((SOWFApower[:, 1]-Power[:, 1])**2)

    if model is 'gauss':
        print 'error_turbine2: ', error_turbine2, 'Dw0: ', prob['model_params:Dw0'], \
            'Angle: ', prob['model_params:rotation_offset_angle'], 'm: ', prob['model_params:m'], \
            'ky: ', prob['model_params:ky']
    elif model is 'floris':
        print 'error_turbine2: ', error_turbine2
        print 'kd: ', xdict['kd'], 'initialWakeAngle: ', xdict['initialWakeAngle'], \
            'initialWakeDisplacement: ', xdict['initialWakeDisplacement'], 'bd: ', xdict['bd'], 'ke: ', xdict['ke'], \
            'me: ', np.array([xdict['me'][0], xdict['me'][1], 1.0]), \
            'MU: ', np.array([xdict['MU'][0], 1.0, xdict['MU'][1]]), 'aU: ', xdict['aU'], 'bU: ', xdict['bU'], \
            'cos_spread: ', xdict['cos_spread']

    funcs = {'obj': error_turbine2}
    fail = False
    return funcs, fail


def set_param_vals(xdict):

    global prob
    global model

    if model is 'gauss':
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
            # quit()
            temp = 0
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
        try:
            prob['model_params:yshift'] = xdict['yshift']
            # print prob['model_params:n_std_dev']
        except:
            # print "here here"
            # quit()
            temp = 0

    elif model is 'floris':
        # set tuning variables
        # prob['gen_params:pP'] = xdict['pP']
        prob['model_params:kd'] = xdict['kd']
        prob['model_params:initialWakeAngle'] = xdict['initialWakeAngle']
        prob['model_params:initialWakeDisplacement'] = xdict['initialWakeDisplacement']
        prob['model_params:bd'] = xdict['bd']
        prob['model_params:ke'] = xdict['ke']
        prob['model_params:me'] = np.array([xdict['me'][0], xdict['me'][1], 1.0])
        prob['model_params:MU'] = np.array([xdict['MU'][0], 1.0, xdict['MU'][1]])
        prob['model_params:aU'] = xdict['aU']
        prob['model_params:bU'] = xdict['bU']
        prob['model_params:cos_spread'] = xdict['cos_spread']


if __name__ == "__main__":

    global model
    model = 'gauss'    # floris or gauss

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
    if model is 'gauss':
        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                                   wake_model=gauss_wrapper, wake_model_options={'nSamples': 0}, datasize=0,
                                   params_IdepVar_func=add_gauss_params_IndepVarComps,
                                   params_IndepVar_args={}))
        prob.setup()
        prob['model_params:integrate'] = False
        prob['model_params:spread_mode'] = 'linear'
        prob['model_params:n_std_dev'] = 4.0
        # prob['model_params:m'] = 0.33
        # prob['model_params:Dw0'] = 1.3
    elif model is 'floris':
        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                                   wake_model=floris_wrapper,
                                   wake_model_options={'nSamples': 0, 'use_rotor_components': False,
                                                       'differentiable': True}, datasize=0,
                                   params_IdepVar_func=add_floris_params_IndepVarComps))
        prob.setup()
        prob['model_params:useWakeAngle'] = True

    # initialize optimization problem
    optProb = Optimization('Tuning %s Model to SOWFA' % model, tuning_obj_function)

    if model is 'gauss':
        optProb.addVarGroup('ke', 1, lower=0.0, upper=1.0, value=0.1, scalar=1E-3)
        # optProb.addVarGroup('spread_angle', 1, lower=0.0, upper=30.0, value=3.0, scalar=1)
        # optProb.addVarGroup('rotation_offset_angle', 1, lower=0.0, upper=50.0, value=1.5, scalar=1E-1)
        # optProb.addVarGroup('ky', 1, lower=0.0, upper=20.0, value=0.1, scalar=1)
        optProb.addVarGroup('Dw0', 3, lower=np.array([0.0, 1.0, 0.0]), upper=np.array([2.9, 1.9, 1.5]), value=np.array([1.3, 1.3, 1.06]))
        #                     scalar=np.ones(3)*1E-2)
        optProb.addVarGroup('m', 3, lower=np.array([0.0, 0.3, -2.]), upper=np.array([0.49, 0.49, 0.]), value=np.array([0.33, 0.33, -0.57]))#, scalar=1E-3)
        optProb.addVarGroup('yshift', 1, lower=-126.4, upper=126.4, value=0.0)#, scalar=1E-3)
    elif model is 'floris':
        # optProb.addVarGroup('pP', 1, lower=0.0, upper=5.0, value=1.5)  # , scalar=1E-1)
        optProb.addVarGroup('kd', 1, lower=0.0, upper=1.0, value=0.15)  # , scalar=1E-1)
        optProb.addVarGroup('initialWakeAngle', 1, lower=-4.0, upper=4.0, value=1.5)  # , scalar=1E-1)
        optProb.addVarGroup('initialWakeDisplacement', 1, lower=-30.0, upper=30.0, value=-4.5)  # , scalar=1E-1)
        optProb.addVarGroup('bd', 1, lower=-1.0, upper=1.0, value=-0.01)  # , scalar=1E-1)
        optProb.addVarGroup('ke', 1, lower=0.0, upper=1.0, value=0.065)  # , scalar=1E-1)
        optProb.addVarGroup('me', 2, lower=np.array([-1.0, 0.0]), upper=np.array([0.0, 0.9]),
                            value=np.array([-0.5, 0.3]))  # , scalar=1E-1)
        optProb.addVarGroup('MU', 2, lower=np.array([0.0, 1.5]), upper=np.array([1.0, 20.0]),
                            value=np.array([0.5, 5.5]))  # , scalar=1E-1)
        optProb.addVarGroup('aU', 1, lower=0.0, upper=20.0, value=5.0)  # , scalar=1E-1)
        optProb.addVarGroup('bU', 1, lower=0.0, upper=5.0, value=1.66)  # , scalar=1E-1)
        optProb.addVarGroup('cos_spread', 1, lower=0.0, upper=10.0, value=2.0)  # , scalar=1E-1)

    # add objective
    optProb.addObj('obj', scale=1E-6)

    # initialize optimizer
    snopt = SNOPT()

    # run optimizer
    sol = snopt(optProb, sens='FD')

    # print solution
    print sol

    # plot fit
    tuning_obj_function(xdict=sol.xStar, plot=True)