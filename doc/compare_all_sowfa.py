import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import cPickle as pickle

from openmdao.api import Problem

from plantenergy.OptimizationGroups import OptAEP
from plantenergy.GeneralWindFarmGroups import AEPGroup
from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps
from plantenergy.floris import floris_wrapper, add_floris_params_IndepVarComps
from plantenergy.utilities import sunflower_points, circumference_points


def plot_data_vs_model(ax=None, datax=np.zeros(0), datay=np.zeros(0), modelx=np.zeros(0),
                       modely=np.zeros(0), title='', xlabel='', ylabel='', datalabel='',
                       modellabel='', modelcolor='r', modelline='--', xscalar=1./126.4, yscalar=1E-3,
                       sum=True, front=True, second=True, legend=True):

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
    if legend:
        ax.legend()
        # quit()
    # quit()
    return ax


def setup_probs():

    nTurbines = 2
    nDirections = 1

    rotorDiameter = 126.4
    rotorArea = np.pi*rotorDiameter*rotorDiameter/4.0
    axialInduction = 1.0/3.0
    # CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CP = 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axialInduction*(1.0-axialInduction)
    generator_efficiency = 0.944

    # Define turbine characteristics
    axialInduction = np.array([axialInduction, axialInduction])
    rotorDiameter = np.array([rotorDiameter, rotorDiameter])
    generatorEfficiency = np.array([generator_efficiency, generator_efficiency])
    yaw = np.array([0., 0.])
    hubHeight = np.array([90.0, 90.0])
    # Define site measurements
    wind_direction = 270.-0.523599*180./np.pi
    wind_speed = 8.    # m/s
    air_density = 1.1716

    Ct = np.array([CT, CT])
    Cp = np.array([CP, CP])
    nRotorPoints = 4
    rotor_pnt_typ = 0
    location = 0.69

    filename = "./input_files/NREL5MWCPCT_dict.p"
    # filename = "../input_files/NREL5MWCPCT_smooth_dict.p"
    import cPickle as pickle

    data = pickle.load(open(filename, "rb"))
    ct_curve = np.zeros([data['wind_speed'].size, 2])
    ct_curve[:, 0] = data['wind_speed']
    ct_curve[:, 1] = data['CT']

    gauss_model_options = {'nSamples': 0,
                          'nRotorPoints': nRotorPoints,
                          'use_ct_curve': True,
                          'ct_curve': ct_curve,
                          'interp_type': 1,
                          'use_rotor_components': False,
                          'verbose': False}

    # gauss_prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
    #                            wake_model=gauss_wrapper, datasize=0, minSpacing=2.0,
    #                            params_IdepVar_func=add_gauss_params_IndepVarComps, wake_model_options=gauss_model_options,
    #                            params_IndepVar_args={'nRotorPoints': nRotorPoints}))

    gauss_prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                                       wake_model=gauss_wrapper, datasize=0,
                                       params_IdepVar_func=add_gauss_params_IndepVarComps,
                                       wake_model_options=gauss_model_options,
                                       params_IndepVar_args={'nRotorPoints': nRotorPoints}))


    floris_options = {'differentiable': True, 'nSamples': 0, 'use_rotor_components': False}

    floris_prob_orig = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=floris_wrapper, wake_model_options=floris_options, datasize=0,
                               params_IdepVar_func=add_floris_params_IndepVarComps,
                               params_IndepVar_args={}))

    floris_prob_tuned = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=floris_wrapper, wake_model_options=floris_options, datasize=0,
                               params_IdepVar_func=add_floris_params_IndepVarComps,
                               params_IndepVar_args={}))

    probs = [gauss_prob, floris_prob_orig, floris_prob_tuned]
    for prob in probs:
        prob.setup()
        if prob is floris_prob_orig or prob is floris_prob_tuned:
            prob['model_params:useWakeAngle'] = True

        turbineX = np.array([1118.1, 1881.9])
        turbineY = np.array([1279.5, 1720.5])
        # prob['gen_params:CTcorrected'] = False
        # prob['gen_params:CPcorrected'] = False
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
        prob['hubHeight'] = hubHeight

        if prob is gauss_prob:

            sort_turbs = True
            wake_combination_method = 1  # can be [0:Linear freestreem superposition,
            #  1:Linear upstream velocity superposition,
            #  2:Sum of squares freestream superposition,
            #  3:Sum of squares upstream velocity superposition]
            ti_calculation_method = 2  # can be [0:No added TI calculations,
            # 1:TI by Niayifar and Porte Agel altered by Annoni and Thomas,
            # 2:TI by Niayifar and Porte Agel 2016,
            # 3:no yet implemented]
            calc_k_star = True
            z_ref = 90.0
            z_0 = 0.001
            k_calc = 0.065

            # tuned with 1 rotor point: error_turbine2:  380593.475508 ky:  0.0147484983033 kz:  0.0365360001244 I:  1.0 shear_exp:  0.0804912726779
            # tuned with 500 rotor points: error_turbine2:  505958.824163 ky:  0.010239469297 kz:  0.0187826477801 I:  0.5 shear_exp:  0.115698347406
            # tuned with 1000 rotor points: error_turbine2:  440240.45048 ky:  0.0132947699754 kz:  0.0267832386866 I:  0.149427342515 shear_exp:  0.107996557048
            # tuned with k_star and 1000 rotor points: error_turbine2:  759565.303289 ky:  0.065 kz:  0.065 I:  0.0765060707278 shear_exp:  0.104381464423
            # using NPA to calculate initial spreading, but then letting BPA adjust it with TI after that. 1000 rotor points
            # error_turbine2:  759565.279351 ky:  0.0330333796913 kz:  0.0330333796913 I:  0.0765060716478 shear_exp:  0.104381467026
            # using NPA to calculate initial spreading, but then letting BPA adjust it with TI after that. 16 rotor points
            # error_turbine2:  642639.730582 ky:  0.0307280539404 kz:  0.0307280539404 I:  0.0704979253074 shear_exp:  0.108435318499
            # tuning only shear_exp with 16 rotor points: error_turbine2:  779216.077341 ky:  0.0267 kz:  0.0267 I:  0.06 shear_exp:  0.161084449732
            I = .063 # + 0.04
            # I = .06
            ky = 0.3837*I + 0.003678
            # ky = 0.022
            kz = 0.3837*I + 0.003678
            # kz = 0.022
            # shear_exp = 0.161084449732
            shear_exp = 0.11

            prob['model_params:wake_combination_method'] = wake_combination_method
            prob['model_params:ti_calculation_method'] = ti_calculation_method
            prob['model_params:calc_k_star'] = calc_k_star
            prob['model_params:sort'] = sort_turbs
            prob['model_params:z_ref'] = z_ref
            prob['model_params:z_0'] = z_0
            prob['model_params:ky'] = ky
            prob['model_params:kz'] = kz
            prob['model_params:I'] = I
            prob['model_params:shear_exp'] = shear_exp
            print "in gauss setup"
            if nRotorPoints > 1:
                if rotor_pnt_typ == 0:
                    points = circumference_points(nRotorPoints, location)
                elif rotor_pnt_typ == 1:
                    points = sunflower_points(nRotorPoints)
                print points
                prob['model_params:RotorPointsY'], prob['model_params:RotorPointsZ'] = points
                print "setting rotor points"

    return probs


# def set_params(probs):

    # floris params
    # probs[2]['model_params:kd'] = 0.224109
    # probs[2]['model_params:initialWakeAngle'] = 3.384485
    # probs[2]['model_params:initialWakeDisplacement'] = 8.407578
    # probs[2]['model_params:bd'] = -0.010000
    # probs[2]['model_params:ke'] = 0.055072
    # probs[2]['model_params:me'] = np.array([-0.000001, 0.181752, 1.0])
    # probs[2]['model_params:MU'] = np.array([0.933389, 1.0, 17.558286])
    # probs[2]['model_params:aU'] = 6.751072
    # probs[2]['model_params:bU'] = 1.681766
    # probs[2]['model_params:cos_spread'] = 9.989090

    # gauss params
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

    # for n_std_dev = 4, error = 1332.49, not integrating, power law
    # probs[0]['model_params:ke'] = 0.051360
    # probs[0]['model_params:rotation_offset_angle'] = 3.197348
    # probs[0]['model_params:Dw0'] = 1.804024
    # probs[0]['model_params:Dw0'] = 1.63
    # probs[0]['model_params:m'] = 0.00

    # for n_std_dev = 5.4, error = 1136.21, not integrating, power law
    # probs[0]['model_params:ke'] = 0.051147
    # probs[0]['model_params:rotation_offset_angle'] = 3.616963
    # probs[0]['model_params:Dw0'] = 1.834599
    # probs[0]['model_params:m'] = 0.096035

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

    # for power law yaw, deficit, and expansion, error = 1759.5
    # probs[0]['model_params:rotation_offset_angle'] = 1.393646
    # probs[0]['model_params:Dw0'] = 1.254036
    # probs[0]['model_params:m'] = 0.166732

    # for power law yaw, deficit, and expansion (reccomended values)
    # probs[0]['model_params:rotation_offset_angle'] = 1.393646
    # probs[0]['model_params:Dw0'] = 1.33
    # probs[0]['model_params:m'] = 0.33

    # for power law all, Dw0 separate, tuned m
    # probs[0]['model_params:rotation_offset_angle'] = 1.454099
    # probs[0]['model_params:Dw0'] = np.array([1.305050, 1.401824, 1.420907])
    # probs[0]['model_params:m'] = 0.101128

    # for power law all, Dw0 separate, constant m
    # probs[0]['model_params:rotation_offset_angle'] = 1.454099
    # probs[0]['model_params:rotation_offset_angle'] = 1.096865
    # probs[0]['model_params:Dw0'] = np.array([1.281240, 0.897360, 0.911161])
    # probs[0]['model_params:Dw0'] = np.array([1.3, 1.00005, 1.])
    # probs[0]['model_params:m'] = 0.

    # for power all but deficit with constant m
    # probs[0]['model_params:ke'] = 0.051126
    # probs[0]['model_params:rotation_offset_angle'] = 3.603684
    # probs[0]['model_params:Dw0'] = np.array([1.794989, 0.863206, 1.])
    # probs[0]['model_params:m'] = 0.33

    # for power law all with constant m
    # probs[0]['model_params:rotation_offset_angle'] = 0.620239
    # probs[0]['model_params:Dw0'] = np.array([1.265505, 0.958504, 0.896609])
    # probs[0]['model_params:Dw0'] = np.array([1.3, 0.958504, 0.896609])
    # probs[0]['model_params:m'] = 0.33

    # for power law all with tuned m
    # probs[0]['model_params:rotation_offset_angle'] = 0.727846
    # probs[0]['model_params:Dw0'] = np.array([1.185009, 1.140757, 1.058244])
    # probs[0]['model_params:m'] = 0.230722

    # for power law all with tuned m and double weight yaw error
    # probs[0]['model_params:rotation_offset_angle'] = 0.802148541875
    # probs[0]['model_params:Dw0'] = np.array([1.18307813, 1.16833547, 1.08521648])
    # probs[0]['model_params:m'] = 0.210864251457

    # for power law all with tuned m and 20x weight yaw error
    # probs[0]['model_params:rotation_offset_angle'] = 0.871926
    # probs[0]['model_params:Dw0'] = np.array([1.190799, 1.223558, 1.142646])
    # probs[0]['model_params:m'] = 0.167548

    # for power law all with individually tuned m and Dw0
    # probs[0]['model_params:rotation_offset_angle'] = 0.811689835284
    # probs[0]['model_params:Dw0'] = np.array([1.22226021, 1.39849858, 0.97207545])
    # probs[0]['model_params:m'] = np.array([0.15566507, 0.1, 0.28422703])

    # for power law all with individually tuned m and Dw0, yaw weighted by 3
    # probs[0]['model_params:rotation_offset_angle'] = 0.884526810188
    # probs[0]['model_params:Dw0'] = np.array([1.21546909, 1.37702043, 0.95703538])
    # probs[0]['model_params:m'] = np.array([0.17499415, 0.1, 0.28738021])

    # for power law all with individually tuned m and Dw0, yaw weighted by 3
    # probs[0]['model_params:rotation_offset_angle'] = 0.726281139043
    # probs[0]['model_params:Dw0'] = np.array([10.80879724, 1.25208657, 0.62180341])
    # probs[0]['model_params:m'] = np.array([0.5014354, 0.1, 0.53332655])

    # for individual power law for diam and deficit. Yaw with linear model
    # probs[0]['model_params:rotation_offset_angle'] = 0.810644329131
    # probs[0]['model_params:Dw0'] = np.array([1.3, 1.64288886, 0.9818137])
    # probs[0]['model_params:m'] = np.array([0.33, 0., 0.27860778])
    # probs[0]['model_params:ky'] = 0.0679899837662

    # for power law all with individually tuned m and Dw0, using 2*a instead of a-1
    # probs[0]['model_params:rotation_offset_angle'] = 2.11916457882
    # probs[0]['model_params:Dw0'] = np.array([1.86868658, 1.6258426, 0.94648549])
    # probs[0]['model_params:m'] = np.array([0., 0., 0.29782246])

    # # for power law with individually tuned m and Dw0, linear yaw, including rotor offset, using 2*a instead of a-1
    # probs[0]['model_params:rotation_offset_angle'] = 1.482520
    # probs[0]['model_params:ky'] = 0.204487
    # probs[0]['model_params:Dw0'] = np.array([1.3, 0.607414, 0.107801])
    # probs[0]['model_params:m'] = np.array([0.33, 0., 0.964934])

    # for power law with individually tuned m and Dw0 including rotor offset for diam and deficit, using 2*a instead of a-1
    # probs[0]['model_params:rotation_offset_angle'] = 2.054952
    # probs[0]['model_params:Dw0'] = np.array([1.869272, 0.612485, 0.123260])
    # probs[0]['model_params:m'] = np.array([0., 0., 0.885561])

    # for power law with individually tuned m and Dw0 using Aitken power law for deficit, linear offset
    # probs[0]['model_params:rotation_offset_angle'] = 0.921858
    # probs[0]['model_params:ky'] = 0.085021
    # probs[0]['model_params:Dw0'] = np.array([1.342291, 1.641186, 0.728072])
    # probs[0]['model_params:m'] = np.array([0.100775, 0., -0.585698])

    # for power law with individually tuned m and Dw0 using Aitken power law for deficit, inflow for Fleming data at 8.3....
    # probs[0]['model_params:rotation_offset_angle'] = 1.062842
    # probs[0]['model_params:rotation_offset_angle'] = 2.062842
    # probs[0]['model_params:Dw0'] = np.array([1.333577, 1.621352, 0.639195])
    # probs[0]['model_params:m'] = np.array([0.130396, 0., -0.522295])

    # for power law with individually tuned m and Dw0 using Aitken power law for deficit, inflow for Fleming data at 8.3....
    # probs[0]['model_params:rotation_offset_angle'] = 0.946076
    # probs[0]['model_params:Dw0'] = np.array([1.353735, 1.623139, 0.656002])
    # probs[0]['model_params:m'] = np.array([0.236072, 0., -0.541287])

    # for power law with suggested m and Dw0 using Aitken power law for deficit, inflow for Fleming data at 8.3....
    # probs[0]['model_params:rotation_offset_angle'] = 1.5
    # probs[0]['model_params:Dw0'] = np.array([1.3, 1.3, 0.56])
    # probs[0]['model_params:m'] = np.array([0.33, 0.33, -0.57])

    # linear everything - coupled - tuned to all data - inflow for Fleming data at 8.3....
    # probs[0]['model_params:ke'] = 0.052166
    # probs[0]['model_params:spread_angle'] = 3.156446
    # probs[0]['model_params:rotation_offset_angle'] = 1.124459
    # probs[0]['model_params:ky'] = 0.247883

    # for n_std_dev = 4, error = 1332.49, not integrating, power law
    # probs[0]['model_params:ke'] = 0.051360
    # probs[0]['model_params:rotation_offset_angle'] = 3.197348
    # probs[0]['model_params:Dw0'] = np.array([1.804024, 1.804024, 1.804024])
    # probs[0]['model_params:m'] = np.array([0.0, 0.0, 0.0])

    # for n_std_dev = 4, linear all, 2*D
    # probs[0]['model_params:ke'] = 0.112334
    # probs[0]['model_params:ky'] = 0.468530
    # probs[0]['model_params:spread_angle'] = 0.0
    # probs[0]['model_params:rotation_offset_angle'] = 1.915430

    # rederived yaw with power. Power law all
    # probs[0]['model_params:rotation_offset_angle'] = 1.5*0.946076
    # probs[0]['model_params:Dw0'] = np.array([1.353735, 1.623139, 0.656002])
    # probs[0]['model_params:m'] = np.array([0.236072, 0.0, -0.541287])

    # rederived yaw with power. Power law all. Dw0[0]=Dw0[1], m[0]=m[1]
    # probs[0]['model_params:rotation_offset_angle'] = 1.02985
    # probs[0]['model_params:Dw0'] = np.array([1.388779, 1.388779, 0.642637])
    # probs[0]['model_params:m'] = np.array([0.100669, 0.100669, -0.530337])

    # rederived yaw with power. Power law all. Dw0[0]=Dw0[1], m[0]=m[1], tuned to all data
    # probs[0]['model_params:rotation_offset_angle'] = 1.052238
    # probs[0]['model_params:Dw0'] = np.array([1.364724, 1.364724, 0.663934])
    # probs[0]['model_params:m'] = np.array([0.092746, 0.092746, -0.542009])

    # rederived yaw with power. Power law all. Dw0[0]=Dw0[1], m[0]=m[1], tuned to all data
    # rederived deficit using actuator disc and momentum balance
    # probs[0]['model_params:rotation_offset_angle'] = 2.089085
    # probs[0]['model_params:Dw0'] = np.array([1.488695, 1.488695, 0.560000])
    # probs[0]['model_params:m'] = np.array([0.000000, 0.000000, -0.542009])

    # probs[0]['model_params:rotation_offset_angle'] = 1.749621
    # probs[0]['model_params:Dw0'] = np.array([1.267740, 1.267740, 0.560000])
    # probs[0]['model_params:m'] = np.array([0.000000, 0.000000, -0.542009])

    # power law as per Aitken et all plus axial induction*2
    # this is a pretty reasonable fit, but defines no expansion in the wake
    # probs[0]['model_params:rotation_offset_angle'] = 2.229160
    # probs[0]['model_params:Dw0'] = np.array([1.889748, 1.603116, 1.037203])
    # probs[0]['model_params:m'] = np.array([0.000000, 0.000000, -0.563005])

    # power law as per Aitken et all plus axial induction*2, added x shift by 1D
    # probs[0]['model_params:rotation_offset_angle'] = 2.078138 + 1.5
    # probs[0]['model_params:Dw0'] = np.array([2.040208, 1.596522, 1.474140])
    # probs[0]['model_params:m'] = np.array([0.00000, 0.000000, -0.698327])

    # power law as per Aitken et all plus axial induction*2, added x shift by 1D except for deficit
    # also a reasonable fit, but no wake expansion
    # probs[0]['model_params:rotation_offset_angle'] = 2.038664
    # probs[0]['model_params:Dw0'] = np.array([2.038664, 1.601559, 1.055975])
    # probs[0]['model_params:m'] = np.array([0.00000, 0.000000, -0.574079])

    # power law as per Aitken et all plus axial induction*2, added y shift tunable
    # excellent fit, but no wake expansion and uses linear yaw offset
    # probs[0]['model_params:rotation_offset_angle'] = 8.466369
    # probs[0]['model_params:Dw0'] = np.array([1.893739, 1.586107, 0.987548])
    # probs[0]['model_params:m'] = np.array([0.00000, 0.000000, -0.524822])
    # probs[0]['model_params:yshift'] = -21.775754

    # probs[0]['model_params:rotation_offset_angle'] = 10.762858
    # probs[0]['model_params:Dw0'] = np.array([1.748372, 1.345945, 1.045982])
    # probs[0]['model_params:m'] = np.array([0.100000, 0.100000, -0.556969])
    # probs[0]['model_params:yshift'] = -30.551647

    #  using Bastankhah with linear yaw
    # probs[0]['model_params:ke'] = 0.077491
    # probs[0]['model_params:ky'] = 0.159944
    # probs[0]['model_params:yshift'] = -4.614311

    # Bastankhah with Bastankhah yaw
    # probs[0]['model_params:ke'] = 0.07747
    # probs[0]['model_params:ky'] = 0.159944
    # probs[0]['model_params:yshift'] = -4.614311

    # probs[0]['model_params:ke'] = 0.078413
    # probs[0]['model_params:ky'] = 0.641951
    # probs[0]['model_params:yshift'] = -3.870224

    # probs[0]['model_params:ke'] = 0.038558
    # probs[0]['model_params:ky'] = 0.078129
    # probs[0]['model_params:yshift'] = -19.948941
    # probs[0]['model_params:rotation_offset_angle'] = -4.0

    # probs[0]['model_params:ke'] = 0.038993
    # probs[0]['model_params:ky'] = 0.087260
    # probs[0]['model_params:yshift'] = -4.614311

    # probs[0]['model_params:ke'] = 0.0390487790134
    # probs[0]['model_params:ky'] = 0.039
    # probs[0]['model_params:rotation_offset_angle'] = 0.72681975016

    # ke = ky tuned to all
    # probs[0]['model_params:ke'] = 0.039166
    # probs[0]['model_params:ky'] = 0.039166
    # probs[0]['model_params:rotation_offset_angle'] = 1.044754

    # ke != ky tuned to all
    # probs[0]['model_params:ke'] = 0.039200
    # probs[0]['model_params:ky'] = 0.048369
    # probs[0]['model_params:rotation_offset_angle'] = 1.175184

    # ke != ky tuned to 7D
    # probs[0]['model_params:ke'] = 0.035706
    # probs[0]['model_params:ky'] = 0.046970
    # probs[0]['model_params:rotation_offset_angle'] = 2.342700

    # ke = ky tuned to 7D
    # probs[0]['model_params:ke'] = 0.036002
    # probs[0]['model_params:ky'] = 0.036002
    # probs[0]['model_params:rotation_offset_angle'] = 1.5

    # Bastankhah with power yaw
    # probs[0]['model_params:ke'] = 0.07747
    # probs[0]['model_params:Dw0'] = np.array([1.49752, 1.3, 1.3])
    # probs[0]['model_params:m'] = np.array([0.23975, 0.33, 0.33])
    # probs[0]['model_params:yshift'] = -4.63626

     # linear everything - coupled - tuned to all data - inflow for Fleming data at 8.3....
    # probs[0]['model_params:ke'] = 0.051690
    # probs[0]['model_params:spread_angle'] = 3.115443
    # probs[0]['model_params:rotation_offset_angle'] = 1.235173
    # probs[0]['model_params:ky'] = 0.205729

    # probs[0]['model_params:integrate'] = False
    # probs[0]['model_params:spread_mode'] = 'power'
    # probs[0]['model_params:yaw_mode'] = 'power'
    # probs[0]['model_params:n_std_dev'] = 4.

if __name__ == "__main__":

    probs = setup_probs()

    # set_params(probs)

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

    print probs[1]['wtVelocity0']
    print probs[1]['wtPower0']
    print probs[1]['AEP']

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
    PFvelocity = 8.38673684
    power_scalar = 1E-3
    distance_scalar = 1./rotor_diameter
    velocity_scalar = 1.
    angle_scalar = 1.
    floris_color = 'b'
    gauss_color = 'r'
    floris_tuned_color = 'g'
    floris_line = '-'
    floris_tuned_line = '-.'
    gauss_line = '--'

    FlorisError = 0.0
    GaussError = 0.0
    FlorisTunedError = 0.0

    # ################## compare yaw ######################
    YawPowFig, YawPowAx = plt.subplots(ncols=2, nrows=1, sharey=False)
    plt.hold(True)
    # 4D yaw
    yawrange = np.array(list(yawrange_4D))
    GaussianPower = list()
    FlorisPower = list()
    FlorisPowerTuned = list()

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
        FlorisPowerTuned.append(list(probs[2]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)
    FlorisPowerTuned = np.array(FlorisPowerTuned)

    # print FlorisPower
    print FlorisPower
    print GaussianPower

    SOWFApower = SOWFApower_yaw_4D*1E-3

    plot_data_vs_model(ax=YawPowAx[0], modelx=yawrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=angle_scalar, yscalar=power_scalar, legend=True)

    plot_data_vs_model(ax=YawPowAx[0], datax=yawrange, datay=SOWFApower, modelx=yawrange,
                       modely=GaussianPower, title='4D', xlabel='yaw angle (deg.)', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=angle_scalar, yscalar=power_scalar, legend=True)

    # plot_data_vs_model(ax=YawPowAx[0], datax=yawrange, datay=SOWFApower, modelx=yawrange,
    #                    modely=FlorisPowerTuned, title='4D', xlabel='yaw angle (deg.)', ylabel='Power (MW)',
    #                    datalabel='SOWFA', modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=angle_scalar, yscalar=power_scalar)
    FlorisError += np.sum((SOWFApower[:, 1]-FlorisPower[:, 1])**2)
    GaussError += np.sum((SOWFApower[:, 1]-GaussianPower[:, 1])**2)
    FlorisTunedError += np.sum((SOWFApower[:, 1]-FlorisPowerTuned[:, 1])**2)

    # 7D yaw
    yawrange = ICOWESdata['yaw'][0]
    GaussianPower = list()
    FlorisPower = list()
    FlorisPowerTuned = list()

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
        FlorisPowerTuned.append(list(probs[2]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)
    FlorisPowerTuned = np.array(FlorisPowerTuned)

    # plot

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0], ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    plot_data_vs_model(ax=YawPowAx[1], modelx=yawrange,
                       modely=FlorisPower,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=angle_scalar, yscalar=power_scalar, legend=True)

    plot_data_vs_model(ax=YawPowAx[1], datax=yawrange, datay=SOWFApower, modelx=yawrange,
                       modely=GaussianPower, title='7D', xlabel='yaw angle (deg.)', ylabel='Power (MW)',
                       datalabel='SOWFA', modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=angle_scalar, yscalar=power_scalar, legend=True)

    # plot_data_vs_model(ax=YawPowAx[1], datax=yawrange, datay=SOWFApower, modelx=yawrange,
    #                    modely=FlorisPowerTuned, title='7D', xlabel='yaw angle (deg.)', ylabel='Power (MW)',
    #                    datalabel='SOWFA', modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=angle_scalar, yscalar=power_scalar)

    FlorisError += np.sum((SOWFApower[:, 1]-FlorisPower[:, 1])**2)
    GaussError += np.sum((SOWFApower[:, 1]-GaussianPower[:, 1])**2)
    FlorisTunedError += np.sum((SOWFApower[:, 1]-FlorisPowerTuned[:, 1])**2)

    # ################## compare position ######################
    PosPowFig, PosPowAx = plt.subplots(ncols=2, nrows=2, sharey=False)

    for prob in probs:
        prob['yaw0'] = np.array([0.0, 0.0])
        prob['windSpeeds'] = np.array([PFvelocity])

    # position crosswind 4D
    posrange = np.array(list(posrange_cs_4D))
    GaussianPower = list()
    FlorisPower = list()
    FlorisPowerTuned = list()

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
        FlorisPowerTuned.append(list(probs[2]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)
    FlorisPowerTuned = np.array(FlorisPowerTuned)

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

    # plot_data_vs_model(ax=PosPowAx[0, 0], datax=posrange, datay=SOWFApower, modelx=posrange,
    #                    modely=FlorisPowerTuned, title='4D', xlabel='y/D', ylabel='Power (MW)',
    #                    datalabel='SOWFA', modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=distance_scalar, yscalar=power_scalar)

    FlorisError += np.sum((SOWFApower[:, 1]-FlorisPower[:, 1])**2)
    GaussError += np.sum((SOWFApower[:, 1]-GaussianPower[:, 1])**2)
    FlorisTunedError += np.sum((SOWFApower[:, 1]-FlorisPowerTuned[:, 1])**2)

    # position crosswind 6D
    posrange = np.array(list(posrange_cs_6D))
    GaussianPower = list()
    FlorisPower = list()
    FlorisPowerTuned = list()

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
        FlorisPowerTuned.append(list(probs[2]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)
    FlorisPowerTuned = np.array(FlorisPowerTuned)

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

    # plot_data_vs_model(ax=PosPowAx[0, 1], datax=posrange, datay=SOWFApower, modelx=posrange,
    #                    modely=FlorisPowerTuned, title='6D', xlabel='y/D', ylabel='Power (MW)',
    #                    datalabel='SOWFA', modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=distance_scalar, yscalar=power_scalar)

    FlorisError += np.sum((SOWFApower[:, 1]-FlorisPower[:, 1])**2)
    GaussError += np.sum((SOWFApower[:, 1]-GaussianPower[:, 1])**2)
    FlorisTunedError += np.sum((SOWFApower[:, 1]-FlorisPowerTuned[:, 1])**2)

    # position crosswind 7D
    posrange = ICOWESdata['pos'][0]
    GaussianPower = list()
    FlorisPower = list()
    FlorisPowerTuned = list()

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
        FlorisPowerTuned.append(list(probs[2]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)
    FlorisPowerTuned = np.array(FlorisPowerTuned)

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

    # plot_data_vs_model(ax=PosPowAx[1, 0], datax=posrange, datay=SOWFApower, modelx=posrange,
    #                    modely=FlorisPowerTuned, title='7D', xlabel='y/D', ylabel='Power (MW)',
    #                    datalabel='SOWFA', modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=distance_scalar, yscalar=power_scalar)

    FlorisError += np.sum((SOWFApower[:, 1]-FlorisPower[:, 1])**2)
    GaussError += np.sum((SOWFApower[:, 1]-GaussianPower[:, 1])**2)
    FlorisTunedError += np.sum((SOWFApower[:, 1]-FlorisPowerTuned[:, 1])**2)

    # position downstream
    posrange = np.array(list(posrange_ds))*rotor_diameter
    GaussianPower = list()
    FlorisPower = list()
    FlorisPowerTuned = list()

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
        FlorisPowerTuned.append(list(probs[2]['wtPower0']))

    GaussianPower = np.array(GaussianPower)
    FlorisPower = np.array(FlorisPower)
    FlorisPowerTuned = np.array(FlorisPowerTuned)

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

    # plot_data_vs_model(ax=PosPowAx[1, 1], datax=posrange, datay=SOWFApower, modelx=posrange,
    #                    modely=FlorisPowerTuned, title='Downstream', xlabel='x/D', ylabel='Power (MW)',
    #                    datalabel='SOWFA', modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=distance_scalar, yscalar=power_scalar)

    FlorisError += np.sum((SOWFApower[:, 1]-FlorisPower[:, 1])**2)
    GaussError += np.sum((SOWFApower[:, 1]-GaussianPower[:, 1])**2)
    FlorisTunedError += np.sum((SOWFApower[:, 1]-FlorisPowerTuned[:, 1])**2)

    print 'Floris error: ', FlorisError, ' Gauss error: ', GaussError, 'Floris Re-Tuned Error: ', FlorisTunedError

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
    FlorisVelocityTuned = list()

    for pos2 in posrange:
        for prob in probs:
            prob['turbineY'] = np.array([0, pos2])
            prob.run()

        GaussianVelocity.append(list(probs[0]['wtVelocity0']))
        FlorisVelocity.append(list(probs[1]['wtVelocity0']))
        FlorisVelocityTuned.append(list(probs[2]['wtVelocity0']))

    FlorisVelocity = np.array(FlorisVelocity)
    GaussianVelocity = np.array(GaussianVelocity)
    FlorisVelocityTuned = np.array(FlorisVelocityTuned)

    plot_data_vs_model(ax=PosVelAx[1, 0], modelx=posrange,
                       modely=FlorisVelocity/PFvelocity,
                       modellabel='FLORIS', modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False)

    plot_data_vs_model(ax=PosVelAx[1, 0], modelx=posrange, modely=GaussianVelocity/PFvelocity, title='7D',
                       xlabel='y/D', ylabel='Velocity (m/s)',
                       modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False)

    # plot_data_vs_model(ax=PosVelAx[1, 0], modelx=posrange, modely=FlorisVelocityTuned/PFvelocity, title='7D',
    #                    xlabel='y/D', ylabel='Velocity (m/s)',
    #                    modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=distance_scalar, yscalar=velocity_scalar, sum=False)
    # plt.legend()
    # plt.show()

    # velocity downstream inline
    posrange = np.linspace(-1.*rotor_diameter, 30.*rotor_diameter, num=1000)

    for prob in probs:
        prob['turbineY'] = np.array([0, 0])

    GaussianVelocity = list()
    FlorisVelocity = list()
    FlorisVelocityTuned = list()

    for pos2 in posrange:

        for prob in probs:
            prob['turbineX'] = np.array([0, pos2])
            prob.run()

        GaussianVelocity.append(list(probs[0]['wtVelocity0']))
        FlorisVelocity.append(list(probs[1]['wtVelocity0']))
        FlorisVelocityTuned.append(list(probs[2]['wtVelocity0']))

    FlorisVelocity = np.array(FlorisVelocity)
    GaussianVelocity = np.array(GaussianVelocity)
    FlorisVelocityTuned = np.array(FlorisVelocityTuned)

    plot_data_vs_model(ax=PosVelAx[1, 1], modelx=posrange,
                       modely=FlorisVelocity/PFvelocity, modellabel='FLORIS',
                       modelcolor=floris_color, modelline=floris_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False, front=False, legend=True)

    plot_data_vs_model(ax=PosVelAx[1, 1], modelx=posrange, modely=GaussianVelocity/PFvelocity, title='Downstream (inline)',
                       xlabel='y/D', ylabel='Velocity (m/s)',
                       modellabel='Gauss', modelcolor=gauss_color, modelline=gauss_line,
                       xscalar=distance_scalar, yscalar=velocity_scalar, sum=False, front=False, legend=True)

    # plot_data_vs_model(ax=PosVelAx[1, 1], modelx=posrange, modely=FlorisVelocityTuned/PFvelocity, title='Downstream (inline)',
    #                    xlabel='y/D', ylabel='Velocity (m/s)',
    #                    modellabel='Floris Re-Tuned', modelcolor=floris_tuned_color,
    #                    modelline=floris_tuned_line, xscalar=distance_scalar, yscalar=velocity_scalar,
    #                    sum=False, front=False, legend=True)

    PosVelAx[1, 1].plot(np.array([7.0, 7.0]), np.array([0.0, 1.2]), ':k', label='Tuning Point')

    plt.xlabel('x/D')
    plt.ylabel('Velocity (m/s)')
    # plt.legend(loc=4,labels=['FLORIS, SOWFA, BPA'])
    plt.show()