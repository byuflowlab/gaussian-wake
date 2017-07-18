import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cPickle as pickle

from openmdao.api import Problem

from pyoptsparse import Optimization, SNOPT

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps
from wakeexchange.utilities import sunflower_points


def tuning_obj_function(xdict={'ky': 0.022, 'kz': 0.022, 'I': 0.06, 'shear_exp': 0.15}, plot=False):

    global prob
    global model
    # prob.setup()
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
    prob['hubHeight'] = rotorDiameter
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
        print 'error_turbine2: ', error_turbine2, 'ky: ', prob['model_params:ky'], \
            'kz: ', prob['model_params:kz'], 'I: ', prob['model_params:I'], \
            'shear_exp: ', prob['model_params:shear_exp']
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
    # I = xdict['I']
    ky = 0.3837 * TI + 0.003678
    kz = 0.3837 * TI + 0.003678

    if model is 'gauss':
        try:
            prob['model_params:ky'] = ky
        except:
            tmp = 0
        try:
            prob['model_params:kz'] = kz
        except:
            tmp = 0
        try:
            prob['model_params:I'] = xdict['I']
        except:
            tmp = 0
        try:
            prob['model_params:shear_exp'] = xdict['shear_exp']
            # print prob['model_params:n_std_dev']
        except:
            tmp = 0
            # quit()
            # raise UserWarning("shear_exp not found")

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
    hub_height = 90.0

    # Define turbine characteristics
    axialInduction = np.array([axialInduction, axialInduction])
    rotorDiameter = np.array([rotorDiameter, rotorDiameter])
    generatorEfficiency = np.array([generator_efficiency, generator_efficiency])
    yaw = np.array([0., 0.])
    hubHeight = np.array([hub_height, hub_height])

    # Define site measurements
    wind_direction = 270.-0.523599*180./np.pi
    wind_speed = 8.    # m/s
    air_density = 1.1716

    Ct = np.array([CT, CT])
    Cp = np.array([CP, CP])

    global prob
    if model is 'gauss':

        sort_turbs = True
        wake_combination_method = 1  # can be [0:Linear freestreem superposition,
        #  1:Linear upstream velocity superposition,
        #  2:Sum of squares freestream superposition,
        #  3:Sum of squares upstream velocity superposition]
        ti_calculation_method = 2  # can be [0:No added TI calculations,
        # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
        # 2:TI by Niayifar and Porte Agel 2016,
        # 3:no yet implemented]
        calc_k_star = False
        z_ref = 90.0
        z_0 = 0.0
        TI = 0.06
        # k_calc = 0.022
        k_calc = 0.3837 * TI + 0.003678
        nRotorPoints = 16

        prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                                   wake_model=gauss_wrapper, wake_model_options={'nSamples': 0}, datasize=0,
                                   params_IdepVar_func=add_gauss_params_IndepVarComps,
                                   params_IndepVar_args={}))

        prob.setup()

        prob['model_params:wake_combination_method'] = wake_combination_method
        prob['model_params:ti_calculation_method'] = ti_calculation_method
        prob['model_params:calc_k_star'] = calc_k_star
        prob['model_params:sort'] = sort_turbs
        prob['model_params:z_ref'] = z_ref
        prob['model_params:z_0'] = z_0
        prob['model_params:ky'] = k_calc
        prob['model_params:kz'] = k_calc
        prob['model_params:I'] = TI

        if nRotorPoints > 1:
            prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = sunflower_points(nRotorPoints)
            print "setting rotor points"

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
        # optProb.addVarGroup('ky', 1, lower=0.01, upper=1.0, value=0.022, scalar=1E1)
        # optProb.addVarGroup('kz', 1, lower=0.01, upper=1.0, value=0.022, scalar=1E1)
        # optProb.addVarGroup('I', 1, lower=0.04, upper=0.5, value=0.06, scalar=1E1)
        optProb.addVarGroup('shear_exp', 1, lower=0.01, upper=1.0, value=0.15, scalar=1)
        # optProb.addVarGroup('yshift', 1, lower=-126.4, upper=126.4, value=0.0)#, scalar=1E-3)
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
    optProb.addObj('obj', scale=1E0)

    # initialize optimizer
    snopt = SNOPT(options={'Print file': 'SNOPT_print_tune_all.out'})

    # run optimizer
    sol = snopt(optProb, sens=None)

    # print solution
    print sol

    # plot fit
    tuning_obj_function(xdict=sol.xStar, plot=True)