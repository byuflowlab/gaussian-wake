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

    rotor_diameter = 126.4
    hub_height = 90.0
    axial_induction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axial_induction*(1.0-axial_induction)
    generator_efficiency = 0.944
    yaw_init = np.array([1.32217615e+01, 1.32140816e+01, 1.32935750e+01, 1.32968679e+01, 7.30138383e-04, 7.32602720e-04])

    # Define turbine characteristics
    axialInduction = np.zeros(nTurbines) + axial_induction
    rotorDiameter = np.zeros(nTurbines) + rotor_diameter
    generatorEfficiency = np.zeros(nTurbines) + generator_efficiency
    yaw = np.zeros(nTurbines) + yaw_init
    Ct = np.zeros(nTurbines) + CT
    Cp = np.zeros(nTurbines) + CP

    # Define site measurements
    nDirections = 1
    wd_polar = 0.523599*180./np.pi
    wind_direction = 270.-wd_polar
    wind_speed = 8.    # m/s
    air_density = 1.1716

    # set up sampling space
    res = 200
    samplesX = np.linspace(min(turbineX)-2.*rotor_diameter, max(turbineX)+2.*rotor_diameter, res)
    samplesY = np.linspace(min(turbineY)-2.*rotor_diameter, max(turbineY)+2.*rotor_diameter, res)
    samplesX, samplesY = np.meshgrid(samplesX, samplesY)
    samplesX = samplesX.flatten()
    samplesY = samplesY.flatten()
    samplesZ = np.ones(res**2)*hub_height

    # initialize problems
    gauss_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=gauss_wrapper, wake_model_options={'nSamples': res**2}, datasize=0,
                               params_IdepVar_func=add_gauss_params_IndepVarComps, force_fd=True,
                               params_IndepVar_args={}))

    floris_prob = Problem(root=OptAEP(nTurbines=nTurbines, nDirections=nDirections, use_rotor_components=False,
                               wake_model=floris_wrapper, wake_model_options=None, datasize=0,
                               params_IdepVar_func=add_floris_params_IndepVarComps,
                               params_IndepVar_args={}))

    probs = [gauss_prob, floris_prob]
    names = ['gauss', 'floris']

    for indx, prob in enumerate(probs):

        prob.setup()

        if names[indx] is 'gauss':
            # print gauss_prob.root.AEPgroup.unknowns.keys()
            gauss_prob['wsPositionX'] = samplesX
            gauss_prob['wsPositionY'] = samplesY
            gauss_prob['wsPositionZ'] = samplesZ

            gauss_prob['model_params:ke'] = 0.052
            gauss_prob['model_params:spread_angle'] = 6.
            gauss_prob['model_params:rotation_offset_angle'] = 2.0

        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['yaw0'] = yaw
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

    plt.figure()

    color = gauss_prob['wsArray0']
    print samplesX.shape, samplesY.shape, color.shape
    samplesX = samplesX.reshape((res, res))
    samplesY = samplesY.reshape((res, res))
    color = color.reshape((res, res))
    plt.pcolormesh(samplesX, samplesY, color, cmap='bwr')
    plt.xlim([min(samplesX.flatten()), max(samplesX.flatten())])
    plt.ylim([min(samplesY.flatten()), max(samplesY.flatten())])

    plt.xlim([0.0, 3000])
    plt.ylim([0.0, 3000])

    yaw = gauss_prob['yaw0'] + wd_polar
    r = 0.5*rotor_diameter
    for turb in range(0, nTurbines):
        x1 = turbineX[turb] + np.sin(yaw[turb]*np.pi/180.)*r
        x2 = turbineX[turb] - np.sin(yaw[turb]*np.pi/180.)*r
        y1 = turbineY[turb] - np.cos(yaw[turb]*np.pi/180.)*r
        y2 = turbineY[turb] + np.cos(yaw[turb]*np.pi/180.)*r
        plt.plot([x1, x2], [y1, y2], 'k', lw=3)
    plt.show()

