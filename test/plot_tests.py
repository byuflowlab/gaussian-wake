"""
Created by Jared J. Thomas, April. 2019.
FLOW Lab
Brigham Young University
"""

import numpy as np

from plantenergy.OptimizationGroups import AEPGroup

from openmdao.api import Problem

from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps

import matplotlib.pyplot as plt

class plotting_tests_wec():

    def __init__(self):

        rotor_diamter = 126.4
        self.rotor_diameter = rotor_diamter
        # define turbine locations in global reference frame
        turbineX = np.array([0.0, 3.*rotor_diamter, 7.*rotor_diamter])
        turbineY = np.array([-2.*rotor_diamter, 2.*rotor_diamter, 0.0])
        hubHeight = np.zeros_like(turbineX)+90.
        # import matplotlib.pyplot as plt
        # plt.plot(turbineX, turbineY, 'o')
        # plt.plot(np.array([0.0, ]))
        # plt.show()

        # initialize input variable arrays
        nTurbines = turbineX.size
        rotorDiameter = np.zeros(nTurbines)
        axialInduction = np.zeros(nTurbines)
        Ct = np.zeros(nTurbines)
        Cp = np.zeros(nTurbines)
        generatorEfficiency = np.zeros(nTurbines)
        yaw = np.zeros(nTurbines)

        # define initial values
        for turbI in range(0, nTurbines):
            rotorDiameter[turbI] = rotor_diamter           # m
            axialInduction[turbI] = 1.0/3.0
            Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
            Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
            generatorEfficiency[turbI] = 0.944
            yaw[turbI] = 0.     # deg.

        # Define flow properties
        nDirections = 1
        wind_speed = 8.0                                # m/s
        air_density = 1.1716                            # kg/m^3
        wind_direction = 270.                           # deg (N = 0 deg., using direction FROM, as in met-mast data)
        wind_frequency = 1.                             # probability of wind in this direction at this speed

        # set up problem

        wake_model_options = {'nSamples': 0}
        prob = Problem(root=AEPGroup(nTurbines=nTurbines, nDirections=nDirections, wake_model=gauss_wrapper,
                                     wake_model_options=wake_model_options, datasize=0, use_rotor_components=False,
                                     params_IdepVar_func=add_gauss_params_IndepVarComps, differentiable=True,
                                     params_IndepVar_args={}))

        # initialize problem
        prob.setup(check=True)

        # assign values to turbine states
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        prob['hubHeight'] = hubHeight
        prob['yaw0'] = yaw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = np.array([wind_speed])
        prob['model_params:z_ref'] = 90.
        prob['air_density'] = air_density
        prob['windDirections'] = np.array([wind_direction])
        prob['windFrequencies'] = np.array([wind_frequency])
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp
        prob['model_params:wec_factor'] = 1.0
        prob['model_params:exp_rate_multiplier'] = 1.0

        # run the problem
        prob.run()

        self.prob = prob
        self.label = ''

    def plot_results(self):
        import matplotlib.pyplot as plt
        plt.plot(self.pos_range/self.rotor_diameter, self.vel_range, label=self.label)
        plt.show()

    def get_velocity(self, wec_diameter_multiplier=1.0, wec_exp_rate_multiplier=1.0, x0=7, x1=7, y0=-4, y1=4):

        if x0 == x1:
            x_range = np.array([x0])
        else:
            x_range = np.linspace(x0*self.rotor_diameter, x1*self.rotor_diameter)
        y_range = np.linspace(y0*self.rotor_diameter, y1*self.rotor_diameter)

        xx, yy = np.meshgrid(x_range, y_range)

        vel = np.zeros_like(xx)
        prob = self.prob
        prob['model_params:wec_factor'] = wec_diameter_multiplier
        prob['model_params:exp_rate_multiplier'] = wec_exp_rate_multiplier

        for i in np.arange(0, xx.shape[0]):
            for j in np.arange(0, xx.shape[1]):
                prob['turbineX'][2] = xx[int(i), int(j)]
                prob['turbineY'][2] = yy[int(i), int(j)]
                prob.run_once()
                vel[int(i), int(j)] = self.prob['wtVelocity0'][2]

        self.vel = vel
        self.xx = xx
        self.yy = yy

        return 0


if __name__ == "__main__":

    xivals = np.arange(1.0, 10.0, .5)
    mytest = plotting_tests_wec()


    for i in np.arange(0, xivals.size):

        mytest.get_velocity(wec_diameter_multiplier=xivals[i]**1, wec_exp_rate_multiplier=xivals[i]**0)
        plt.plot(mytest.yy/mytest.rotor_diameter, mytest.vel, label=xivals[i])

    plt.ylabel('Inflow Velocity (m/s)')
    plt.xlabel('Y/D')
    plt.legend()
    plt.show()