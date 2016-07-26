import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cPickle as pickle

from openmdao.api import Problem

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


def plot_data_vs_model(ax=None, datax=np.zeros(0), datay=np.zeros(0), modelx=np.zeros(0),
                       modely=np.zeros(0), title='', xlabel='', ylabel='', datalabel='',
                       modellabel='', modelcolor='r', modelline='--', xscalar=1./126.4, yscalar=1E-3,
                       sum=True, front=True, second=True):

    if ax is None:
        fig = plt.figure()
        ax = fig.gca()

    # plot data
    if datax.size > 0:
        if front:
            ax.plot(datax*xscalar, datay[:, 0]*yscalar, 'o', mec='k', mfc='none', label=datalabel)
        if second:
            ax.plot(datax*xscalar, datay[:, 1]*yscalar, '^', mec='k', mfc='none')
        if sum:
            ax.plot(datax*xscalar, datay[:, 0]*yscalar+datay[:, 1]*yscalar, 'ks', mec='k', mfc='none')

    # plot model
    if modelx.size > 0:
        # plot model
        if front:
            ax.plot(modelx*xscalar, modely[:, 0]*yscalar, modelline+modelcolor, label=modellabel)
        if second:
            ax.plot(modelx*xscalar, modely[:, 1]*yscalar, modelline+modelcolor)
        if sum:
            ax.plot(modelx*xscalar, modely[:, 0]*yscalar+modely[:, 1]*yscalar, modelline+'k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    return ax


def setup_probs():

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

    return probs


def set_params(probs):

    # probs[0]['model_params:ke'] = 0.052
    # probs[0]['model_params:spread_angle'] = 6.
    # probs[0]['model_params:rotation_offset_angle'] = 2.0

    # for axialInd calc only
    # probs[0]['model_params:ke'] = 0.050688
    # probs[0]['model_params:spread_angle'] = 7.562716
    # probs[0]['model_params:rotation_offset_angle'] = 3.336568

    # for axialInd and inflow adjust
    # probs[0]['model_params:ke'] = 0.052333
    # probs[0]['model_params:spread_angle'] =  8.111330
    # probs[0]['model_params:rotation_offset_angle'] = 2.770265

    # for inflow adjust only
    # probs[0]['model_params:ke'] = 0.052230
    # probs[0]['model_params:spread_angle'] =  6.368191
    # probs[0]['model_params:rotation_offset_angle'] = 1.855112

    # for added n_st_dev param #1
    # probs[0]['model_params:ke'] = 0.050755
    # probs[0]['model_params:spread_angle'] = 11.205766#*0.97
    # probs[0]['model_params:rotation_offset_angle'] = 3.651790
    # probs[0]['model_params:n_std_dev'] = 9.304371

    # for added n_st_dev param #2
    # probs[0]['model_params:ke'] = 0.051010
    # probs[0]['model_params:spread_angle'] = 11.779591
    # probs[0]['model_params:rotation_offset_angle'] = 3.564547
    # probs[0]['model_params:n_std_dev'] = 9.575505

    # for decoupled ky with n_std_dev = 4
    # probs[0]['model_params:ke'] = 0.051145
    # probs[0]['model_params:spread_angle'] = 2.617982
    # probs[0]['model_params:rotation_offset_angle'] = 3.616082
    # probs[0]['model_params:ky'] = 0.211496

    # for integrating for decoupled ky with n_std_dev = 4, linear, integrating
    # probs[0]['model_params:ke'] = 0.016969
    # probs[0]['model_params:spread_angle'] = 0.655430
    # probs[0]['model_params:rotation_offset_angle'] = 3.615754
    # probs[0]['model_params:ky'] = 0.195392

    # for integrating for decoupled ky with n_std_dev = 4, linear, integrating
    # probs[0]['model_params:ke'] = 0.008858
    # probs[0]['model_params:spread_angle'] = 0.000000
    # probs[0]['model_params:rotation_offset_angle'] = 4.035276
    # probs[0]['model_params:ky'] = 0.199385

    # for decoupled ke with n_std_dev=4, linear, not integrating
    # probs[0]['model_params:ke'] = 0.051190
    # probs[0]['model_params:spread_angle'] = 2.619202
    # probs[0]['model_params:rotation_offset_angle'] = 3.629337
    # probs[0]['model_params:ky'] = 0.211567

    # for decoupled ky with n_std_dev = 4, error = 1332.49, not integrating, power law
    probs[0]['model_params:ke'] = 0.051360
    probs[0]['model_params:rotation_offset_angle'] = 3.197348
    probs[0]['model_params:Dw0'] = 1.804024
    probs[0]['model_params:m'] = 0.0

    # for decoupled ky with n_std_dev = 4, error = 1630.8, with integrating, power law
    # probs[0]['model_params:ke'] = 0.033165
    # probs[0]['model_params:rotation_offset_angle'] = 3.328051
    # probs[0]['model_params:Dw0'] = 1.708328
    # probs[0]['model_params:m'] = 0.0

    # for decoupled ky with n_std_dev = 4, error = 1140.59, not integrating, power law for expansion,
    # linear for yaw
    # probs[0]['model_params:ke'] = 0.050741
    # probs[0]['model_params:rotation_offset_angle'] = 3.628737
    # probs[0]['model_params:Dw0'] = 0.846582
    # probs[0]['model_params:ky'] = 0.207734

    # for decoupled ky with n_std_dev = 4, error = 1058.73, integrating, power law for expansion,
    # linear for yaw
    # probs[0]['model_params:ke'] = 0.016129
    # probs[0]['model_params:rotation_offset_angle'] = 3.644356
    # probs[0]['model_params:Dw0'] = 0.602132
    # probs[0]['model_params:ky'] = 0.191178

    probs[0]['model_params:integrate'] = False
    probs[0]['model_params:spread_mode'] = 'power'
    probs[0]['model_params:n_std_dev'] = 4


if __name__ == "__main__":

    probs = setup_probs()

    set_params(probs)

    # time the models
    import time
    t1 = time.time()
    for i in range(0, 100):
        probs[0].run()
    t2 = time.time()
    for i in range(0, 100):
        probs[1].run()
    t3 = time.time()
    # gauss time:  0.0580031871796
    # floris time:  0.10697388649

    print 'gauss time: ', t2-t1
    print 'floris time: ', t3-t2

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

    # set plot params
    rotor_diameter = probs[0]['rotorDiameter'][0]
    ICOWESvelocity = 8.0
    PFvelocity = 8.48673684
    power_scalar = 1E-3
    distance_scalar = 1./rotor_diameter
    velocity_scalar = 1.
    angle_scalar = 1.
    floris_color = 'b'
    gauss_color = 'r'
    floris_line = '-'
    gauss_line = '--'

    # ################## compare yaw ######################
    YawPowFig, YawPowAx = plt.subplots(ncols=2, nrows=1, sharey=False)
    plt.hold(True)
    # 4D yaw
    yawrange = np.array(list(yawrange_4D))
    GaussianPower = list()
    FlorisPower = list()

    # set to 4D positions and inflow velocity
    for prob in probs:
        prob['turbineX'] = np.array([1118.1, 1556.0])
        prob['turbineY'] = np.array([1279.5, 1532.3])
        prob['windSpeeds'] = np.array([PFvelocity])

    for yaw1 in yawrange:

        for prob in probs:
            prob['yaw0'] = np.array([yaw1, 0.0])
            prob.run()

        GaussianPower.append(list(probs[0]['wtPower0']))
        FlorisPower.append(list(probs[1]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    # print FlorisPower

    SOWFApower = SOWFApower_yaw_4D*1E-3

    plot_data_vs_model(ax=YawPowAx[0], modelx=yawrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=angle_scalar, yscalar=power_scalar)

    plot_data_vs_model(ax=YawPowAx[0], datax=yawrange, datay=SOWFApower, modelx=yawrange,
                       modely=GaussianPower, title='4D', xlabel='yaw angle (deg.)', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=angle_scalar, yscalar=power_scalar)

    # YawPowAx[0].legend(loc=4)
    # 7D yaw
    yawrange = ICOWESdata['yaw'][0]
    GaussianPower = list()
    FlorisPower = list()

    # set to 7D positions
    for prob in probs:
        prob['turbineX'] = np.array([1118.1, 1881.9])
        prob['turbineY'] = np.array([1279.5, 1720.5])
        prob['windSpeeds'] = np.array([ICOWESvelocity])

    # run analysis
    for yaw1 in yawrange:

        for prob in probs:
            prob['yaw0'] = np.array([yaw1, 0.0])
            prob.run()

        GaussianPower.append(list(probs[0]['wtPower0']))
        FlorisPower.append(list(probs[1]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    # plot

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    plot_data_vs_model(ax=YawPowAx[1], modelx=yawrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=angle_scalar, yscalar=power_scalar)

    plot_data_vs_model(ax=YawPowAx[1], datax=yawrange, datay=SOWFApower, modelx=yawrange,
                       modely=GaussianPower, title='7D', xlabel='yaw angle (deg.)', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=angle_scalar, yscalar=power_scalar)

    # ################## compare position ######################
    PosPowFig, PosPowAx = plt.subplots(ncols=2, nrows=2, sharey=False)

    for prob in probs:
        prob['yaw0'] = np.array([0.0, 0.0])
        prob['windSpeeds'] = np.array([PFvelocity])

    # position crosswind 4D
    posrange = np.array(list(posrange_cs_4D))
    print posrange
    GaussianPower = list()
    FlorisPower = list()

    for pos2 in posrange:
        # Define turbine locations and orientation (4D)
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1556.0])
        Yinit = np.array([1279.5, 1532.3])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))
        for prob in probs:
            prob['turbineX'] = XY[0, :]
            prob['turbineY'] = XY[1, :]
            prob.run()

        GaussianPower.append(list(probs[0]['wtPower0']))
        FlorisPower.append(list(probs[1]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    SOWFApower = SOWFApower_cs_4D*1E-3
    # print error_turbine2

    plot_data_vs_model(ax=PosPowAx[0, 0], modelx=posrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    plot_data_vs_model(ax=PosPowAx[0, 0], datax=posrange, datay=SOWFApower, modelx=posrange,
                       modely=GaussianPower, title='4D', xlabel='y/D', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    # position crosswind 6D
    posrange = np.array(list(posrange_cs_6D))
    print posrange
    GaussianPower = list()
    FlorisPower = list()

    for prob in probs:
        prob['windSpeeds'] = np.array([PFvelocity])

    for pos2 in posrange:
        # Define turbine locations and orientation (4D)
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1556.0])
        Yinit = np.array([1279.5, 1532.3])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                        [np.sin(effUdXY), np.cos(effUdXY)]]),
                                               np.array([[0., 0], [0, pos2]]))
        for prob in probs:
            prob['turbineX'] = XY[0, :]
            prob['turbineY'] = XY[1, :]
            prob.run()

        GaussianPower.append(list(probs[0]['wtPower0']))
        FlorisPower.append(list(probs[1]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    SOWFApower = SOWFApower_cs_6D*1E-3
    # print error_turbine2

    plot_data_vs_model(ax=PosPowAx[0, 1], modelx=posrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    plot_data_vs_model(ax=PosPowAx[0, 1], datax=posrange, datay=SOWFApower, modelx=posrange,
                       modely=GaussianPower, title='6D', xlabel='y/D', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    # position crosswind 7D
    posrange = ICOWESdata['pos'][0]
    GaussianPower = list()
    FlorisPower = list()

    for prob in probs:
        prob['windSpeeds'] = np.array([ICOWESvelocity])

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

        GaussianPower.append(list(probs[0]['wtPower0']))
        FlorisPower.append(list(probs[1]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    # print error_turbine2

    plot_data_vs_model(ax=PosPowAx[1, 0], modelx=posrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    plot_data_vs_model(ax=PosPowAx[1, 0], datax=posrange, datay=SOWFApower, modelx=posrange,
                       modely=GaussianPower, title='7D', xlabel='y/D', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    # position downstream
    posrange = np.array(list(posrange_ds))*rotor_diameter
    print posrange
    GaussianPower = list()
    FlorisPower = list()

    for prob in probs:
        prob['windSpeeds'] = np.array([PFvelocity])
        prob['turbineY'] = np.array([0.0, 0.0])
        prob['windDirections'] = np.array([270.0])

    for pos2 in posrange:

        for prob in probs:
            prob['turbineX'] = np.array([0.0, pos2])
            prob.run()

        GaussianPower.append(list(probs[0]['wtPower0']))
        FlorisPower.append(list(probs[1]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)

    SOWFApower = SOWFApower_ds*1E-3
    # print error_turbine2

    plot_data_vs_model(ax=PosPowAx[1, 1], modelx=posrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=power_scalar)

    plot_data_vs_model(ax=PosPowAx[1, 1], datax=posrange, datay=SOWFApower, modelx=posrange,
                       modely=GaussianPower, title='Downstream', xlabel='x/D', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=power_scalar)


    # ################## compare velocity ######################
    PosVelFig, PosVelAx = plt.subplots(ncols=2, nrows=2, sharey=False)

    # velocity crosswind 7D
    posrange = np.linspace(-3.*rotor_diameter, 3.*rotor_diameter, num=1000)

    for prob in probs:
        prob['yaw0'] = np.array([0.0, 0.0])
        prob['windDirections'] = np.array([270.])
        prob['turbineX'] = np.array([0, 7.*rotor_diameter])

    GaussianVelocity = list()
    FlorisVelocity = list()

    for pos2 in posrange:
        for prob in probs:
            prob['turbineY'] = np.array([0, pos2])
            prob.run()

        GaussianVelocity.append(list(probs[0]['wtVelocity0']))
        FlorisVelocity.append(list(probs[1]['wtVelocity0']))

    FlorisVelocity = np.array(FlorisVelocity)
    GaussianVelocity = np.array(GaussianVelocity)

    plot_data_vs_model(ax=PosVelAx[1, 0], modelx=posrange,
                       modely=FlorisVelocity,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False)

    plot_data_vs_model(ax=PosVelAx[1, 0], modelx=posrange, modely=GaussianVelocity, title='7D',
                       xlabel='y/D', ylabel='Velocity (m/s)',
                       modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False)
    # plt.legend()
    # plt.show()

    # velocity downstream inline
    posrange = np.linspace(-1.*rotor_diameter, 30.*rotor_diameter, num=1000)

    for prob in probs:
        prob['turbineY'] = np.array([0, 0])

    GaussianVelocity = list()
    FlorisVelocity = list()

    for pos2 in posrange:

        for prob in probs:
            prob['turbineX'] = np.array([0, pos2])
            prob.run()

        GaussianVelocity.append(list(probs[0]['wtVelocity0']))
        FlorisVelocity.append(list(probs[1]['wtVelocity0']))

    FlorisVelocity = np.array(FlorisVelocity)
    GaussianVelocity = np.array(GaussianVelocity)

    plot_data_vs_model(ax=PosVelAx[1, 1], modelx=posrange,
                       modely=FlorisVelocity, modellabel='FLORIS',
                       modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False, front=False)

    plot_data_vs_model(ax=PosVelAx[1, 1], modelx=posrange, modely=GaussianVelocity, title='Downstream (inline)',
                       xlabel='y/D', ylabel='Velocity (m/s)',
                       modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False, front=False)

    PosVelAx[1, 1].plot(np.array([7.0, 7.0]), np.array([0.0, 9.0]), ':k', label='Tuning Point')

    plt.xlabel('x/D')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc=4)
    plt.show()