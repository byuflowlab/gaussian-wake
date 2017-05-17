import numpy as np
from scipy.integrate import quad
from scipy.io import loadmat
import pylab as plt
import time

from openmdao.api import Component, Problem, Group

from _porteagel_fortran import porteagel_analyze as porteagel_analyze_fortran
from _porteagel_fortran import porteagel_analyze_bv

# def porteagel_analyze_fortran(turbineXw, turbineYw, turbineZ, rotorDiameter,
#                         Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I):
#
#     velocitiesTurbines = _porteagel_analyze(turbineXw, turbineYw, turbineZ, rotorDiameter,
#                         Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I)
#     # print velocitiesTurbines, 'here'
#     return velocitiesTurbines

def wake_offset_func(turbineXw, position_x,rotorDiameter,Ct,yaw, ky, kz, alpha, beta, I):
    #yaw = yaw * np.pi / 180.   # use if yaw is passed in as degrees
    x0 = rotorDiameter * (np.cos(yaw) * (1.0 + np.sqrt(1.0 - Ct)) /
                                (np.sqrt(2.0) * (alpha * I + beta * (1.0 - np.sqrt(1.0 - Ct)))))
    theta_c_0 = 0.3 * yaw * (1.0 - np.sqrt(1.0 - Ct * np.cos(yaw))) / np.cos(yaw)

    deltax0 = position_x - (turbineXw + x0)

    sigmay = rotorDiameter * (ky * deltax0 / rotorDiameter
                                    + np.cos(yaw) / np.sqrt(8.0))
    sigmaz = rotorDiameter * (kz * deltax0 / rotorDiameter
                                    + 1.0 / np.sqrt(8.0))
    wake_offset = rotorDiameter * (
        theta_c_0 * x0 / rotorDiameter +
        (theta_c_0 / 14.7) * np.sqrt(np.cos(yaw) / (ky * kz * Ct)) *
        (2.9 + 1.3 * np.sqrt(1.0 - Ct) - Ct) *
        np.log(
            ((1.6 + np.sqrt(Ct)) *
             (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                            (np.cos(yaw) * rotorDiameter ** 2))
              - np.sqrt(Ct))) /
            ((1.6 - np.sqrt(Ct)) *
             (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                            (np.cos(yaw) * rotorDiameter ** 2))
              + np.sqrt(Ct)))
        )
    )
    return wake_offset

def porteagel_analyze(nTurbines, turbineXw, turbineYw, turbineZ, rotorDiameter,
                        Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I):

    for i in range(0, nTurbines):
        if (Ct[i] > 0.96):  # Glauert condition
            axialInduction[i] = 0.143 + np.sqrt(0.0203 - 0.6427 * (0.889 - Ct[i]))
        else:
            axialInduction[i] = 0.5 * (1.0 - np.sqrt(1.0 - Ct[i]))

    # NOTE: Bastankhah and Porte Agel 2016 defines yaw as positive clockwise, the negative below accounts for this
    yaw *= -np.pi / 180.

    velocitiesTurbines = np.tile(wind_speed, nTurbines)

    for turb in range(0, nTurbines):
        x0 = rotorDiameter[turb] * (np.cos(yaw[turb]) * (1.0 + np.sqrt(1.0 - Ct[turb])) /
                                    (np.sqrt(2.0) * (alpha * I + beta * (1.0 - np.sqrt(1.0 - Ct[turb])))))
        theta_c_0 = 0.3 * yaw[turb] * (1.0 - np.sqrt(1.0 - Ct[turb] * np.cos(yaw[turb]))) / np.cos(yaw[turb])

        for turbI in range(0, nTurbines):  # at turbineX-locations

            deltax0 = turbineXw[turbI] - (turbineXw[turb] + x0)

            if deltax0 > 0.0:
                sigmay = rotorDiameter[turb] * (ky * deltax0 / rotorDiameter[turb]
                                                + np.cos(yaw[turb]) / np.sqrt(8.0))
                sigmaz = rotorDiameter[turb] * (kz * deltax0 / rotorDiameter[turb]
                                                + 1.0 / np.sqrt(8.0))
                wake_offset = rotorDiameter[turb] * (
                    theta_c_0 * x0 / rotorDiameter[turb] +
                    (theta_c_0 / 14.7) * np.sqrt(np.cos(yaw[turb]) / (ky * kz * Ct[turb])) *
                    (2.9 + 1.3 * np.sqrt(1.0 - Ct[turb]) - Ct[turb]) *
                    np.log(
                        ((1.6 + np.sqrt(Ct[turb])) *
                         (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                                        (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
                          - np.sqrt(Ct[turb]))) /
                        ((1.6 - np.sqrt(Ct[turb])) *
                         (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                                        (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
                          + np.sqrt(Ct[turb])))
                    )
                )
                # print wake_offset, turbineYw[turbI]

                deltay = turbineYw[turbI] - (turbineYw[turb] + wake_offset)

                deltav = wind_speed * (
                    (1.0 - np.sqrt(1.0 - Ct[turb] *
                                   np.cos(yaw[turb]) / (8.0 * sigmay * sigmaz /
                                                        (rotorDiameter[turb] ** 2)))) *
                    np.exp(-0.5 * ((deltay) / sigmay) ** 2) *
                    np.exp(-0.5 * ((turbineZ[turbI] - turbineZ[turb]) / sigmaz) ** 2)
                )

                velocitiesTurbines[turbI] -= deltav

    return velocitiesTurbines


def porteagel_visualize(nTurbines, nSamples, turbineXw, turbineYw, turbineZ, velX, velY, velZ, rotorDiameter,
                        Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I):

    for i in range(0, nTurbines):
        if (Ct[i] > 0.96):  # Glauert condition
            axialInduction[i] = 0.143 + np.sqrt(0.0203 - 0.6427 * (0.889 - Ct[i]))
        else:
            axialInduction[i] = 0.5 * (1.0 - np.sqrt(1.0 - Ct[i]))

    # NOTE: Bastankhah and Porte Agel 2016 defines yaw as positive clockwise, the negative below accounts for this
    yaw *= -np.pi / 180.
    ws_array = np.tile(wind_speed, nSamples)

    for turb in range(0, nTurbines):
        x0 = rotorDiameter[turb] * (np.cos(yaw[turb]) * (1.0 + np.sqrt(1.0 - Ct[turb])) /
                                    (np.sqrt(2.0) * (alpha * I + beta * (1.0 - np.sqrt(1.0 - Ct[turb])))))
        theta_c_0 = 0.3 * yaw[turb] * (1.0 - np.sqrt(1.0 - Ct[turb] * np.cos(yaw[turb]))) / np.cos(yaw[turb])

        for loc in range(0, nSamples):  # at velX-locations
            deltax0 = velX[loc] - (turbineXw[turb] + x0)
            if deltax0 > 0.0:
                sigmay = rotorDiameter[turb] * (ky * deltax0 / rotorDiameter[turb]
                                                + np.cos(yaw[turb]) / np.sqrt(8.0))
                sigmaz = rotorDiameter[turb] * (kz * deltax0 / rotorDiameter[turb]
                                                + 1.0 / np.sqrt(8.0))
                wake_offset = rotorDiameter[turb] * (
                    theta_c_0 * x0 / rotorDiameter[turb] +
                    (theta_c_0 / 14.7) * np.sqrt(np.cos(yaw[turb]) / (ky * kz * Ct[turb])) *
                    (2.9 + 1.3 * np.sqrt(1.0 - Ct[turb]) - Ct[turb]) *
                    np.log(
                        ((1.6 + np.sqrt(Ct[turb])) *
                         (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                                        (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
                          - np.sqrt(Ct[turb]))) /
                        ((1.6 - np.sqrt(Ct[turb])) *
                         (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                                        (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
                          + np.sqrt(Ct[turb])))
                    )
                )

                deltay = velY[loc] - (turbineYw[turb] + wake_offset)

                deltav = wind_speed * (
                    (1.0 - np.sqrt(1.0 - Ct[turb] *
                                   np.cos(yaw[turb]) / (8.0 * sigmay * sigmaz /
                                                        (rotorDiameter[turb] ** 2)))) *
                    np.exp(-0.5 * ((deltay) / sigmay) ** 2) *
                    np.exp(-0.5 * ((velZ[loc] - turbineZ[turb]) / sigmaz) ** 2)
                )

                ws_array[loc] -= deltav

    return ws_array


class GaussianWake(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(GaussianWake, self).__init__()

        self.deriv_options['type'] = 'user'
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 1.0e-12
        # self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        if options is None:
            self.radius_multiplier = 1.0
            self.nSamples = nSamples = 0
        else:
            # self.radius_multiplier = options['radius multiplier']
            self.nSamples = nSamples = options['nSamples']

        # unused but required for compatibility

        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))

        # used
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('hubHeight', val=np.zeros(nTurbines), units='m')
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('Ct', np.zeros(nTurbines), desc='Turbine thrust coefficients')
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)

        # options
        # self.add_param('language', val='fortran')

        # params for Bastankhah with yaw
        self.add_param('model_params:ky', val=0.022, pass_by_object=True)
        self.add_param('model_params:kz', val=0.022, pass_by_object=True)
        self.add_param('model_params:alpha', val=2.32, pass_by_object=True)
        self.add_param('model_params:beta', val=0.154, pass_by_object=True)
        self.add_param('model_params:I', val=0.075, pass_by_object=True, desc='turbulence intensity')\

        self.add_output('wtVelocity%i' % direction_id, val=np.zeros(nTurbines), units='m/s')

        if nSamples > 0:
            # visualization input
            self.add_param('wsPositionXw', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='downwind position of desired measurements in wind ref. frame')
            self.add_param('wsPositionYw', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='crosswind position of desired measurements in wind ref. frame')
            self.add_param('wsPositionZ', np.zeros(nSamples), units='m', pass_by_object=True,
                           desc='position of desired measurements in wind ref. frame')

            # visualization output
            self.add_output('wsArray%i' % direction_id, np.zeros(nSamples), units='m/s', pass_by_object=True,
                            desc='wind speed at measurement locations')

    def solve_nonlinear(self, params, unknowns, resids):

        nTurbines = self.nTurbines
        direction_id = self.direction_id
        nSamples = self.nSamples

        # params for Bastankhah model with yaw
        ky = params['model_params:ky']
        kz = params['model_params:kz']
        alpha = params['model_params:alpha']
        beta = params['model_params:beta']
        I = params['model_params:I']

        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['hubHeight']
        yaw = params['yaw%i' % direction_id]
        rotorDiameter = params['rotorDiameter']
        Ct = params['Ct']
        axialInduction = params['axialInduction']
        wind_speed = params['wind_speed']

        # run the Bastankhah and Porte Agel model
        # velocitiesTurbines = porteagel_analyze(nTurbines=nTurbines, turbineXw=turbineXw, turbineYw=turbineYw,
        #                                        turbineZ=turbineZ, rotorDiameter=rotorDiameter, Ct=Ct,
        #                                        axialInduction=axialInduction, wind_speed=wind_speed, yaw=np.copy(yaw),
        #                                        ky=ky, kz=kz, alpha=alpha, beta=beta, I=I)

        velocitiesTurbines = porteagel_analyze_fortran(turbineXw, turbineYw,
                                               turbineZ, rotorDiameter, Ct,
                                               wind_speed, np.copy(yaw),
                                               ky, kz, alpha, beta, I)

        # velocitiesTurbines = _porteagel_analyze(turbineXw, turbineYw, turbineZ, rotorDiameter,
        #                    Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I)
        # print 'velocities: ', velocitiesTurbines
        unknowns['wtVelocity%i' % direction_id] = velocitiesTurbines

        if nSamples > 0.0:

            velX = params['wsPositionXw']
            velY = params['wsPositionYw']
            velZ = params['wsPositionZ']

            ws_array = porteagel_visualize(nTurbines, nSamples, turbineXw, turbineYw, turbineZ, velX, velY, velZ, rotorDiameter,
                                           Ct, axialInduction, wind_speed, np.copy(yaw), ky, kz, alpha, beta, I)

            unknowns['wsArray%i' % direction_id] = ws_array


    def linearize(self, params, unknowns, resids):

        # obtain id for this wind direction
        direction_id = self.direction_id

        # x and y positions w.r.t. the wind dir. (wind dir. = +x)
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['hubHeight']

        # yaw wrt wind dir. (wind dir. = +x)
        yawDeg = params['yaw%i' % self.direction_id]

        # turbine specs
        rotorDiameter = params['rotorDiameter']

        # air flow
        wind_speed = params['wind_speed']
        Ct = params['Ct']

        # wake model parameters
        ky = params['model_params:ky']
        kz = params['model_params:kz']
        alpha = params['model_params:alpha']
        beta = params['model_params:beta']
        I = params['model_params:I']

        # define jacobian size
        nTurbines = len(turbineXw)
        nDirs = nTurbines

        # define input array to direct differentiation
        wtVelocityb = np.eye(nDirs, nTurbines)

        # call to fortran code to obtain output values
        turbineXwb, turbineYwb, turbineZb, rotorDiameterb, Ctb, yawDegb, _ = \
            porteagel_analyze_bv(turbineXw, turbineYw, turbineZ,
                                 rotorDiameter, Ct, wind_speed, yawDeg,
                                 ky, kz, alpha, beta, I,
                                 wtVelocityb)

        # initialize Jacobian dict
        J = {}

        # collect values of the Jacobian
        J['wtVelocity%i' % direction_id, 'turbineXw'] = turbineXwb
        J['wtVelocity%i' % direction_id, 'turbineYw'] = turbineYwb
        J['wtVelocity%i' % direction_id, 'hubHeight'] = turbineZb
        J['wtVelocity%i' % direction_id, 'yaw%i' % direction_id] = yawDegb
        J['wtVelocity%i' % direction_id, 'rotorDiameter'] = rotorDiameterb
        J['wtVelocity%i' % direction_id, 'Ct'] = Ctb
        # print J
        return J


if __name__ == "__main__":

    nTurbines = 2
    nDirections = 1

    rotor_diameter = 126.4
    rotorArea = np.pi*rotor_diameter*rotor_diameter/4.0
    axialInduction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    # CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axialInduction*(1.0-axialInduction)
    generator_efficiency = 0.944

    # Define turbine characteristics
    axialInduction = np.array([axialInduction, axialInduction])
    rotorDiameter = np.array([rotor_diameter, rotor_diameter])
    # rotorDiameter = np.array([rotorDiameter, 0.0001*rotorDiameter])
    yaw = np.array([0., 0.])

    # Define site measurements
    wind_direction = 30.
    wind_speed = 8.    # m/s
    air_density = 1.1716

    Ct = np.array([CT, CT])
    Cp = np.array([CP, CP])

    turbineX = np.array([0.0, 7.*rotor_diameter])
    turbineY = np.array([0.0, 0.0])

    prob = Problem()
    prob.root = Group()
    prob.root.add('model', GaussianWake(nTurbines), promotes=['*'])

    prob.setup()

    prob['turbineXw'] = turbineX
    prob['turbineYw'] = turbineY

    GaussianWakeVelocity = list()

    yawrange = np.linspace(-40., 40., 400)

    for yaw1 in yawrange:

        prob['yaw0'] = np.array([yaw1, 0.0])
        prob['Ct'] = Ct*np.cos(prob['yaw0']*np.pi/180.)**2

        prob.run()

        velocitiesTurbines = prob['wtVelocity0']

        GaussianWakeVelocity.append(list(velocitiesTurbines))

    GaussianWakeVelocity = np.array(GaussianWakeVelocity)

    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)
    axes[0, 0].plot(yawrange, GaussianWakeVelocity[:, 0]/wind_speed, 'b')
    axes[0, 0].plot(yawrange, GaussianWakeVelocity[:, 1]/wind_speed, 'b')

    axes[0, 0].set_xlabel('yaw angle (deg.)')
    axes[0, 0].set_ylabel('Velcoity ($V_{eff}/V_o$)')

    posrange = np.linspace(-3.*rotor_diameter, 3.*rotor_diameter, 100)

    prob['yaw0'] = np.array([0.0, 0.0])

    GaussianWakeVelocity = list()

    for pos2 in posrange:

        prob['turbineYw'] = np.array([0.0, pos2])

        prob.run()

        velocitiesTurbines = prob['wtVelocity0']

        GaussianWakeVelocity.append(list(velocitiesTurbines))

    GaussianWakeVelocity = np.array(GaussianWakeVelocity)

    wind_speed = 1.0
    axes[0, 1].plot(posrange/rotor_diameter, GaussianWakeVelocity[:, 0]/wind_speed, 'b')
    axes[0, 1].plot(posrange/rotor_diameter, GaussianWakeVelocity[:, 1]/wind_speed, 'b')
    axes[0, 1].set_xlabel('y/D')
    axes[0, 1].set_ylabel('Velocity ($V_{eff}/V_o$)')

    posrange = np.linspace(-3.*rotorDiameter[0], 3.*rotorDiameter[0], num=1000)

    posrange = np.linspace(-1.*rotorDiameter[0], 30.*rotorDiameter[0], num=2000)
    yaw = np.array([0.0, 0.0])
    wind_direction = 0.0

    GaussianWakeVelocity = list()
    for pos2 in posrange:

        prob['turbineXw'] = np.array([0.0, pos2])
        prob['turbineYw'] = np.array([0.0, 0.0])

        prob.run()

        velocitiesTurbines = prob['wtVelocity0']

        GaussianWakeVelocity.append(list(velocitiesTurbines))

    GaussianWakeVelocity = np.array(GaussianWakeVelocity)

    axes[1, 1].plot(posrange/rotor_diameter, GaussianWakeVelocity[:, 1], 'y', label='GH')
    axes[1, 1].plot(np.array([7, 7]), np.array([2, 8]), '--k', label='tuning point')

    plt.xlabel('x/D')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc=4)

    plt.show()