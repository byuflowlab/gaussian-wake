"""
Created by Jared J. Thomas, April. 2019.
FLOW Lab
Brigham Young University
"""

import numpy as np

from plantenergy.OptimizationGroups import AEPGroup

from openmdao.api import Problem

from plantenergy.gauss import gauss_wrapper, add_gauss_params_IndepVarComps

from _porteagel_fortran import wake_offset_func

import matplotlib.pyplot as plt

class plotting_tests_wec():

    def __init__(self):

        rotor_diameter = 126.4
        self.rotor_diameter = rotor_diameter
        # define turbine locations in global reference frame
        turbineX = np.array([0.0, 3.*rotor_diameter, 7.*rotor_diameter])
        turbineY = np.array([-2.*rotor_diameter, 2.*rotor_diameter, 0.0])
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
            rotorDiameter[turbI] = rotor_diameter           # m
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
                                     params_IdepVar_args={}))

        # initialize problem
        prob.setup(check=False)

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
        prob['model_params:wec_spreading_angle'] = 0.0
        prob['model_params:wake_model_version'] = '2016'
        prob['model_params:ti_calculation_method'] = 4
        prob['model_params:I'] = 0.1
        prob['model_params:WECH'] = 0

        # run the problem
        prob.run_driver()

        self.prob = prob
        self.label = ''

    def plot_results(self):
        import matplotlib.pyplot as plt
        plt.plot(self.pos_range/self.rotor_diameter, self.vel_range, label=self.label)
        plt.show()

    def get_velocity(self, wec_diameter_multiplier=1.0, wec_wec_spreading_angle=0.0, x0=4.5, x1=4.5, y0=-4, y1=4, res=50):

        if x0 == x1:
            x_range = np.array([x0*self.rotor_diameter])
        else:
            x_range = np.linspace(x0*self.rotor_diameter, x1*self.rotor_diameter, res)
        y_range = np.linspace(y0*self.rotor_diameter, y1*self.rotor_diameter, res)

        xx, yy = np.meshgrid(x_range, y_range)

        vel = np.zeros_like(xx)
        prob = self.prob
        prob['model_params:wec_factor'] = wec_diameter_multiplier
        prob['model_params:wec_spreading_angle'] = wec_wec_spreading_angle

        for i in np.arange(0, xx.shape[0]):
            for j in np.arange(0, xx.shape[1]):
                prob['turbineX'][2] = xx[int(i), int(j)]
                prob['turbineY'][2] = yy[int(i), int(j)]
                # print( prob['turbineX'], prob['turbineY'])
                prob.run_model()
                vel[int(i), int(j)] = self.prob['wtVelocity0'][2]

        self.vel = vel
        self.xx = xx
        self.yy = yy

        return 0

    def get_aep(self, wec_diameter_multiplier=1.0, wec_wec_spreading_angle=0.0, x0=4.5, x1=4.5, y0=-4, y1=4, res=50):

        if x0 == x1:
            x_range = np.array([x0*self.rotor_diameter])
        else:
            x_range = np.linspace(x0*self.rotor_diameter, x1*self.rotor_diameter, res)
        y_range = np.linspace(y0*self.rotor_diameter, y1*self.rotor_diameter, res)

        xx, yy = np.meshgrid(x_range, y_range)

        aep = np.zeros_like(xx)
        prob = self.prob
        prob['model_params:wec_factor'] = wec_diameter_multiplier
        prob['model_params:wec_spreading_angle'] = wec_wec_spreading_angle

        for i in np.arange(0, xx.shape[0]):
            for j in np.arange(0, xx.shape[1]):
                prob['turbineX'][2] = xx[int(i), int(j)]
                prob['turbineY'][2] = yy[int(i), int(j)]
                # print( prob['turbineX'], prob['turbineY'])
                prob.run_model()
                aep[int(i), int(j)] = self.prob['AEP']

        self.aep = aep
        self.xx = xx
        self.yy = yy

        return 0

    def plot_cross_sections(self, exp_type='angle', save_image=True):

        if exp_type == 'angle':
            # xivals = np.arange(1.0, 100.0, 5)
            xivals = np.array([1.0, 10, 20, 30, 40, 50, 60, 70])
            angle_on = 1
            diam_on = 0
        elif exp_type =="diam":
            xivals = np.arange(1.0, 10.0, .5)
            diam_on = 1
            angle_on = 0
        elif exp_type == "hybrid":
            xivals = np.arange(1.0, 10.0, .5)
            diam_on = 1
            angle_on = 0
            self.prob['model_params:WECH'] = 1
        else:
            raise ValueError("incorrect value specified for exp_type in plot_cross_sections")

        for i in np.arange(0, xivals.size):
            self.get_velocity(wec_diameter_multiplier=xivals[i] ** diam_on, wec_wec_spreading_angle=xivals[i] ** angle_on)
            plt.plot(self.yy / self.rotor_diameter, self.vel, label=xivals[i])

        plt.ylabel('Inflow Velocity (m/s)')
        plt.xlabel('Y/D')
        plt.legend(loc=3, frameon=False)
        if save_image:
            plt.savefig(exp_type+''+".pdf",tranparent=True)
        plt.show()

    def plot_contour(self, exp_type='angle', xival=0., save_fig=False):

        if exp_type == 'angle':
            angle_on = 1
            diam_on = 0
        elif exp_type =="diam":
            diam_on = 1
            angle_on = 0
        elif exp_type == "hybrid":
            diam_on = 1
            angle_on = 0
            self.prob['model_params:WECH'] = 1
        else:
            raise ValueError("incorrect value specified for exp_type in plot_contour")


        # self.get_velocity(wec_diameter_multiplier=xival ** diam_on, wec_wec_spreading_angle=xival ** angle_on, x0=0., x1=10.)
        self.get_aep(wec_diameter_multiplier=xival ** diam_on, wec_wec_spreading_angle=xival * angle_on, x0=0., x1=10.)


        # plt.contourf(self.xx/self.rotor_diameter, self.yy / self.rotor_diameter, self.vel, cmap='coolwarm')
        plt.contourf(self.xx/self.rotor_diameter, self.yy / self.rotor_diameter, self.aep, cmap='coolwarm')

        plt.ylabel('Y/D')
        plt.xlabel('X/D')
        plt.legend()
        if save_fig:
            plt.title('%s Factor: %s' %(exp_type, xival))
            plt.savefig('%s_%.2f.pdf' % (exp_type, xival))
        else:
            plt.show()

    def plot_data(self):

        data = np.genfromtxt('./input_files/wpd_datasets.csv',delimiter=',',usecols=np.arange(0,8),skip_header=True)
        data0x = data[~np.isnan(data[:,0]), 0]
        data0y = data[~np.isnan(data[:, 1]), 1]
        data1x = data[~np.isnan(data[:, 2]), 2]
        data1y = data[~np.isnan(data[:, 3]), 3]
        data2x = data[~np.isnan(data[:, 4]), 4]
        data2y = data[~np.isnan(data[:, 5]), 5]
        data3x = data[~np.isnan(data[:, 6]), 6]
        data3y = data[~np.isnan(data[:, 7]), 7]

        data0x, data0y = zip(*sorted(zip(data0x, data0y)))
        data1x, data1y = zip(*sorted(zip(data1x, data1y)))
        data2x, data2y = zip(*sorted(zip(data2x, data2y)))
        data3x, data3y = zip(*sorted(zip(data3x, data3y)))

        plt.plot([0, data0x[-1], data0x[-1]], [data0y[0], data0y[0], 0])
        plt.plot([0, data1x[-1], data1x[-1]], [data1y[0], data1y[0], 0])
        plt.plot(data2x, data2y)
        plt.plot(data3x, data3y)

        plt.show()

        return 0

    def plot_data_with_model(self):

        # data
        data = np.genfromtxt('./input_files/wpd_datasets.csv', delimiter=',', usecols=np.arange(0, 8), skip_header=True)
        data0x = data[~np.isnan(data[:, 0]), 0]
        data0y = data[~np.isnan(data[:, 1]), 1]
        data1x = data[~np.isnan(data[:, 2]), 2]
        data1y = data[~np.isnan(data[:, 3]), 3]
        data2x = data[~np.isnan(data[:, 4]), 4]
        data2y = data[~np.isnan(data[:, 5]), 5]
        data3x = data[~np.isnan(data[:, 6]), 6]
        data3y = data[~np.isnan(data[:, 7]), 7]

        data0x, data0y = np.array(zip(*sorted(zip(data0x, data0y))))
        data1x, data1y = np.array(zip(*sorted(zip(data1x, data1y))))
        data2x, data2y = np.array(zip(*sorted(zip(data2x, data2y))))
        data3x, data3y = np.array(zip(*sorted(zip(data3x, data3y))))

        # model
        from _porteagel_fortran import point_velocity_with_shear_func

        turbI = -1
        wake_combination_method = 1
        wake_model_version = 2016
        sorted_x_idx = np.array([0])
        pointY = 0.0
        pointZ = 0.125
        tol = 1E-12
        alpha = 2.32
        beta = 0.154
        expratemultiplier = 0.0
        wec_factor = 1.0
        wind_speed = 4.88
        z_ref = 0.125
        z_0 = 0.000022
        shear_exp = 0.1
        turbineXw = np.array([0])
        turbineYw = np.array([0])
        turbineZ = np.array([0.125])
        rotorDiameter = np.array([0.15])
        yaw = np.array([20.*np.pi/180.0])
        wtVelocity = np.array([wind_speed])
        Ct_local = 0.7361200568897026*np.ones_like(turbineXw) #np.array([0.7374481936835376])
        TIturbs = 0.08*np.ones_like(turbineXw) #*np.array([0.01]) #np.array([0.001])
        ky_local = 0.022 #np.array([0.3837*TIturbs[0] + 0.003678])
        kz_local = 0.022 #np.array([0.3837*TIturbs[0] + 0.003678])

        # a = (1./(2.*np.cos(yaw)))*(1.-np.sqrt(1.-Ct_local*np.cos(yaw)))
        Ct_local[:] = np.cos(yaw)*Ct_local
        print( Ct_local)
        # quit()
        def x0_func(yaw, ct, alpha, beta, ti, d):

            x0od = np.cos(yaw)*(1.+np.sqrt(1.-ct))/(np.sqrt(2.)*(alpha*ti+beta*(1.-np.sqrt(1.-ct))))

            return x0od

        x0od = x0_func(yaw[0], Ct_local[0], alpha, beta, TIturbs[0], rotorDiameter[0])
        print( 'test x0/d is ', x0od)
        print( 'exp x0/d is', data0x[-1])
        # quit()
        modelx = np.linspace(tol*1E3, 20, 100)
        modely = np.linspace(0, 0, 1)
        point_vel = np.ones([modelx.size, modely.size])
        for i in np.arange(0, modelx.size):
            pointX = modelx[i] * rotorDiameter[0]
            # print( 'pointX (python) = ', pointX)
            _, wake_offset = point_velocity_with_shear_func(turbI, wake_combination_method,
                                                            wake_model_version,
                                                            sorted_x_idx, pointX, pointY, pointZ,
                                                            tol, alpha, beta, expratemultiplier, wec_factor,
                                                            wind_speed, z_ref, z_0, shear_exp,
                                                            turbineXw, turbineYw, turbineZ,
                                                            rotorDiameter, yaw, wtVelocity,
                                                            Ct_local, TIturbs, ky_local, kz_local)
            print( 'offset 1', wake_offset)

            for j in np.arange(0, modely.size):

                pointY = wake_offset + modely[j]*rotorDiameter[0]
                # pointY = modely[j]*rotorDiameter[0]
                point_vel[i, j], wake_offset = point_velocity_with_shear_func(turbI, wake_combination_method,
                                                                wake_model_version,
                                                                sorted_x_idx, pointX, pointY, pointZ,
                                                                tol, alpha, beta, expratemultiplier, wec_factor,
                                                                wind_speed, z_ref, z_0, shear_exp,
                                                                turbineXw, turbineYw, turbineZ,
                                                                rotorDiameter, yaw, wtVelocity,
                                                                Ct_local, TIturbs, ky_local, kz_local)

                # print( 'offset 2', point_vel[i], wake_offset)

        modelval = (wind_speed-point_vel)/wind_speed
        # plot


        plt.plot([data0x[-1], data0x[-1]], [0, 1], '-.r', label='Experimental $x_0$')
        # plt.plot([0, data1x[-1], data1x[-1]], [data1y[0], data1y[0], 0],'--', label='Experimental $x_0$, for #\gamma_f$')
        plt.plot(data2x, data2y,'-r', label='Experimental Deficit')
        # plt.plot(data3x, data3y,'--')
        plt.plot([x0od, x0od], [0, 1], '--b', label='Our $x_0$')

        for k in np.arange(0, modely.size):
            plt.plot(modelx, modelval[:, k], 'b', label='Our Deficit') #label='%.2f' % (modely[k]))
        plt.ylabel('$max[\Delta u / u_h]$')
        plt.xlabel('$x/d$')
        # plt.title('Using $x_0$ as published')
        plt.title('Using $x_0$ with postitive sign in numerator')
        plt.legend()
        plt.show()

        return 0


class bpa_wind_tunnel_plots():
    def __init__(self):
        self.tolerance = 1E-2
        self.turbI = -1
        self.wake_combination_method = 1
        self.wake_model_version = 2016
        self.sorted_x_idx = np.array([0])
        self.rotorDiameter = np.array([0.15])
        self.pointX = self.rotorDiameter * 5.
        self.pointY = 0.24 * self.rotorDiameter
        self.pointZ = 0.125
        self.tol = 1E-12
        self.alpha = 2.32
        self.beta = 0.154
        self.expratemultiplier = 0.0
        self.wec_factor = 1.0
        self.wind_speed = 4.88
        self.z_ref = 0.125
        self.z_0 = 0.000022
        self.shear_exp = 0.1
        self.turbineXw = np.array([0])
        self.turbineYw = np.array([0])
        self.turbineZ = np.array([0.125])
        self.yaw = np.array([20. * np.pi / 180.0])
        self.wtVelocity = np.array([self.wind_speed])
        self.Ct_local = 0.7361200568897026 * np.ones_like(self.turbineXw)  # np.array([0.7374481936835376])
        self.TIturbs = 0.074 * np.ones_like(self.turbineXw)  # *np.array([0.01]) #np.array([0.001])
        self.ky_local = 0.022  # np.array([0.3837*TIturbs[0] + 0.003678])
        self.kz_local = 0.022  # np.array([0.3837*TIturbs[0] + 0.003678])

    def plot_cross_wind_profile(self):
        from _porteagel_fortran import point_velocity_with_shear_func
        npoints = 100

        ypoints = np.linspace(-2, 2, npoints)*self.rotorDiameter
        velocity = np.zeros_like(ypoints)

        for i in np.arange(0, ypoints.size):
            velocity[i] = point_velocity_with_shear_func(self.turbI, self.wake_combination_method,
                                                                   self.wake_model_version,
                                                                   self.sorted_x_idx, self.pointX, ypoints[i],
                                                                   self.pointZ, self.tol,
                                                                   self.alpha, self.beta, self.expratemultiplier,
                                                                   self.wec_factor,
                                                                   self.wind_speed, self.z_ref, self.z_0,
                                                                   self.shear_exp, self.turbineXw,
                                                                   self.turbineYw, self.turbineZ, self.rotorDiameter,
                                                                   self.yaw,
                                                                   self.wtVelocity, self.Ct_local, self.TIturbs,
                                                                   self.ky_local,
                                                                   self.kz_local)

        data = np.loadtxt('input_files/bpa_windtunnel_cross_5d_yaw0.txt', delimiter=',')

        plt.plot(data[:,0], data[:,1], 'o', label='published bp model')
        plt.plot(1-velocity/self.wind_speed, ypoints/self.rotorDiameter, label='our bp model')
        plt.xlabel('Wind speed deficit percentage')
        plt.ylabel('y/d')
        plt.legend()
        plt.show()

if __name__ == "__main__":

    mytest = plotting_tests_wec()
    # mytest.plot_data_with_model()
    # mytest.plot_cross_sections(exp_type='hybrid')
    # mytest.plot_contour(exp_type='angle', xival=0.0, save_fig=False)
    xis = np.array([0., 10., 20., 30., 40., 50., 60.])
    xis = np.array([1., 2., 4., 6., 8., 10., 12.])
    xis = np.array([1., 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    for xival in xis:
        print( xival)
        mytest.plot_contour(exp_type='hybrid', xival=xival, save_fig=False)


    # mytest = bpa_wind_tunnel_plots()
    # mytest.plot_cross_wind_profile()