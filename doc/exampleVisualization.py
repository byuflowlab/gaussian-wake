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

    # theta = np.arctan((turbineY[5]-turbineY[0])/(turbineX[5]-turbineX[0]))*180./np.pi
    #
    # print theta
    # quit()

    rotor_diameter = 126.4
    hub_height = 90.0
    axial_induction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axial_induction*(1.0-axial_induction)
    generator_efficiency = 0.944
    # yaw_init_gauss = np.array([1.32217615e+01, 1.32140816e+01, 1.32935750e+01, 1.32968679e+01, 7.30138383e-04, 7.32602720e-04])
    # for tuning with n_std_dev #1
    # yaw_init_gauss = 0*np.array([1.26140798e+01, 1.26123736e+01, 1.17078481e+01, 1.17089869e+01, -1.22503914e-07,  -1.31933646e-06])
    yaw_init_gauss = 0*np.array([13.74293166, 13.74283174, 14.8380152, 14.83806931, 0., 0.])
    yaw_init_gauss = np.array([19.00, 19.00, 23.80, 23.80, 0.05, 0.00])

    # for tuning with n_std_dev #2
    # yaw_init_gauss = np.array([1.29093304e+01, 1.29077015e+01, 1.19463208e+01, 1.19474608e+01, 5.61165053e-07, 8.01354835e-07])

    yaw_init_floris = np.array([1.56406883e+01, 1.56406883e+01, 1.73139554e+01, 1.73139554e+01, 1.49265921e-05, 1.49265921e-05])

    # Define turbine characteristics
    axialInduction = np.zeros(nTurbines) + axial_induction
    rotorDiameter = np.zeros(nTurbines) + rotor_diameter
    generatorEfficiency = np.zeros(nTurbines) + generator_efficiency
    yaw_floris = np.zeros(nTurbines) + yaw_init_floris
    yaw_gauss = np.zeros(nTurbines) + yaw_init_gauss
    Ct = np.zeros(nTurbines) + CT
    Cp = np.zeros(nTurbines) + CP

    # Define site measurements
    nDirections = 1
    wd_polar = 0.523599*180./np.pi

    wind_direction = 270.-wd_polar
    wind_speed = 8.    # m/s
    air_density = 1.1716

    # set up sampling space
    res = 400
    # samplesX = np.linspace(min(turbineX)-2.*rotor_diameter, max(turbineX)+2.*rotor_diameter, res)
    # samplesY = np.linspace(min(turbineY)-2.*rotor_diameter, max(turbineY)+2.*rotor_diameter, res)

    samplesX = np.linspace(0, 3000, res)
    samplesY = np.linspace(0, 3000, res)

    samplesX, samplesY = np.meshgrid(samplesX, samplesY)
    samplesX = samplesX.flatten()
    samplesY = samplesY.flatten()
    samplesZ = np.ones(samplesX.shape)*hub_height

    # initialize problems
    gauss_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                         wake_model=gauss_wrapper, wake_model_options={'nSamples': res**2}, datasize=0,
                         params_IdepVar_func=add_gauss_params_IndepVarComps, force_fd=True,
                         params_IndepVar_args={}))

    wake_model_options = {'differentiable': True, 'use_rotor_components': False, 'nSamples': res**2, 'verbose': False}
    floris_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                          wake_model=floris_wrapper, wake_model_options=wake_model_options, datasize=0,
                          differentiable=True, params_IdepVar_func=add_floris_params_IndepVarComps,
                          params_IndepVar_args={}))

    probs = [gauss_prob, floris_prob]
    names = ['gauss', 'floris']

    for indx, prob in enumerate(probs):

        prob.setup()

        if names[indx] is 'gauss':
            # print gauss_prob.root.AEPgroup.unknowns.keys()

            # gauss_prob['model_params:ke'] = 0.052
            # gauss_prob['model_params:spread_angle'] = 6.
            # gauss_prob['model_params:rotation_offset_angle'] = 2.0

            # gauss_prob['model_params:ke'] = 0.050755
            # gauss_prob['model_params:spread_angle'] = 11.205766
            # gauss_prob['model_params:rotation_offset_angle'] = 3.651790
            # gauss_prob['model_params:n_std_dev'] = 9.304371

            # gauss_prob['model_param s:ke'] = 0.051028
            # gauss_prob['model_params:spread_angle'] = 11.862988
            # gauss_prob['model_params:rotation_offset_angle'] = 3.594340
            # gauss_prob['model_params:n_std_dev'] = 12.053127

            # using ky with n_std_dev = 6
            # gauss_prob['model_params:ke'] = 0.051115
            # gauss_prob['model_params:spread_angle'] = 5.967284
            # gauss_prob['model_params:rotation_offset_angle'] = 3.597926
            # gauss_prob['model_params:ky'] = 0.494776

            # using ky with n_std_dev = 3
            # gauss_prob['model_params:ke'] = 0.051079
            # gauss_prob['model_params:spread_angle'] = 0.943942
            # gauss_prob['model_params:rotation_offset_angle'] = 3.579857
            # gauss_prob['model_params:ky'] = 0.078069

            # for decoupled ky with n_std_dev = 4
            gauss_prob['model_params:ke'] = 0.051145
            gauss_prob['model_params:spread_angle'] = 2.617982
            gauss_prob['model_params:rotation_offset_angle'] = 3.616082
            gauss_prob['model_params:ky'] = 0.211496

            # for decoupled ky with n_std_dev = 6 and double diameter wake at rotor pos
            gauss_prob['model_params:ke'] = 0.051030
            gauss_prob['model_params:spread_angle'] = 1.864696
            gauss_prob['model_params:rotation_offset_angle'] = 3.362729
            gauss_prob['model_params:ky'] = 0.193011

            # for integrating for decoupled ky with n_std_dev = 4, error = 1034.3
            gauss_prob['model_params:ke'] = 0.007523
            gauss_prob['model_params:spread_angle'] = 1.876522
            gauss_prob['model_params:rotation_offset_angle'] = 3.633083
            gauss_prob['model_params:ky'] = 0.193160

            prob['yaw0'] = yaw_gauss

        else:
            prob['yaw0'] = yaw_floris

        prob['wsPositionX'] = np.copy(samplesX)
        prob['wsPositionY'] = np.copy(samplesY)
        prob['wsPositionZ'] = np.copy(samplesZ)

        # print prob['wsPositionX'].shape, prob['wsPositionY'], prob['wsPositionZ']
        # quit()

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

        prob.run()

    for indx, prob in enumerate(probs):
        print names[indx], prob['yaw0']

    samplesX = samplesX.reshape((res, res))
    samplesY = samplesY.reshape((res, res))

    for indx, prob in enumerate(probs):
        print names[indx], prob['wtPower0'], np.sum(prob['wtPower0']), min(prob['wsArray0'])
        plt.figure()

        color = prob['wsArray0']
        # color[color==wind_speed] *= 0
        # print color[color>wind_speed]
        # print samplesX.shape, samplesY.shape, color.shape

        color = color.reshape((res, res))

        # color maps that work better: jet, gnuplot2, coolwarm
        plt.pcolormesh(samplesX, samplesY, color, cmap='coolwarm', vmin=1.75, vmax=11.)
        plt.xlim([min(samplesX.flatten()), max(samplesX.flatten())])
        plt.ylim([min(samplesY.flatten()), max(samplesY.flatten())])

        plt.xlim([0.0, 3000])
        plt.ylim([0.0, 3000])

        yaw = prob['yaw0'] + wd_polar
        r = 0.5*rotor_diameter
        for turb in range(0, nTurbines):
            x1 = turbineX[turb] + np.sin(yaw[turb]*np.pi/180.)*r
            x2 = turbineX[turb] - np.sin(yaw[turb]*np.pi/180.)*r
            y1 = turbineY[turb] - np.cos(yaw[turb]*np.pi/180.)*r
            y2 = turbineY[turb] + np.cos(yaw[turb]*np.pi/180.)*r
            plt.plot([x1, x2], [y1, y2], 'k', lw=3)

        plt.title(names[indx])
        plt.axis('square')

    plt.show()

