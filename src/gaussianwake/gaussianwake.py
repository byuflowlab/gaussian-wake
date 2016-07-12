import numpy as np
import scipy as sp
from scipy.io import loadmat
import pylab as plt
import time

from openmdao.api import Component, Problem, Group

def GaussianMax(x, max, mu, sigma):
    """
    Computes the average value of the standard normal distribution on the interval [p1, p2]
    :param max: max value (peak) of the gaussian normal distribution
    :param mu: mean of the standard normal distribution
    :param sigma: standard distribution of the standard normal distribution
    :param p1: first point of the interval of interest
    :param p2: second point of the interval of interest
    :return: average value of the standard normal distribution on given interval
    """

    f = max*np.exp((-(x-mu)**2)/(2.*sigma**2))

    return f


def GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.5, 0.22, 0.725083, 0.5, 1.0, 0.804150])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""

    # xdict = np.array(xdict)
    # print xdict
    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    # me = np.array([-0.5, 0.22, 1.0])*np.array([1., 1., 0.75])
    # MU = np.array([0.5, 1.0, 5.5])*np.array([0., 0., 0.1475])
    me = xdict['xvars'][0:3][2]
    MU = xdict['xvars'][3:6][2]
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180.        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction

    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0, 0])
        velY = np.zeros([0, 0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix, np.array([turbineX, turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size > 0:
        locations = np.dot(rotationMatrix, np.array([velX, velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations, nTurbines))
    wakeCentersYT = np.zeros((nTurbines, nTurbines))
    for turb in range(0, nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]), 2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0, nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb], 0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc, turb] = turbineY[turb]
            wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + \
                (wakeAngleInit*(15*np.power(factor, 4)+np.power(wakeAngleInit, 2))/((30*kd*np.power(factor, 5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit, 4))/(30*kd))  # yaw-induced deflection
        for turbI in range(0, nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI, turb] = turbineY[turb]
            wakeCentersYT[turbI, turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI, turb] = wakeCentersYT[turbI,turb] + \
                (wakeAngleInit*(15*np.power(factor, 4)+np.power(wakeAngleInit, 4))/((30*kd*np.power(factor, 5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit, 4))/(30*kd))  # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations, nTurbines))
    wakeDiametersT = np.zeros((nTurbines, nTurbines))
    for turb in range(0, nTurbines):
        for loc in range(0, nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            wakeDiameters[loc, turb] = rotorDiameter[turb]+2*ke*me*np.maximum(deltax, 0)
        for turbI in range(0, nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            wakeDiametersT[turbI, turb] = np.maximum(rotorDiameter[turb]+2*ke*me*deltax, 0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    # wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    # wakeOverlapTRel = wakeOverlapT
    # for turb in range(0,nTurbines):
    #     wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    # keUncorrected = ke
    # ke = np.zeros(nTurbines)
    # for turb in range(0,nTurbines):
    #     s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
    #     ke[turb] = keUncorrected*(1+s*keCorrDA)

    ke = np.ones(nTurbines)*ke

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc, turb]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor * \
                    np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU)*np.maximum(0, deltax))), 2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0, deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf, nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0, nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]

            if deltax > 0:
                # deltay = wakeCentersYT[turbI, turb] - turbineY[turbI]
                mU = MU / np.cos(aU*np.pi/180. + bU*yaw[turb])
                sigma = (wakeDiametersT[turbI, turb] + rotorDiameter[turbI])/6.
                mu = wakeCentersYT[turbI, turb]
                max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)
                # p1 = wakeCentersYT[turbI, turb] + deltay - rotorDiameter[turbI]/2.
                # p2 = wakeCentersYT[turbI, turb] + deltay + rotorDiameter[turbI]/2.

                # wakeEffCoeffPerZone = GaussianAverage(max, mu, sigma, p1, p2)
                wakeEffCoeffPerZone = GaussianMax(turbineY[turbI], max, mu, sigma)
                wakeEffCoeff = wakeEffCoeff + np.power(axialInd[turb]*wakeEffCoeffPerZone, 2.0)


        wakeEffCoeff = (1. - 2. * np.sqrt(wakeEffCoeff))
        # print 'wakeEffCoeff original ', wakeEffCoeff
        # multiply the inflow speed with the wake coefficients to find effective wind speed at turbine
        velocitiesTurbines[turbI] *= wakeEffCoeff
    # print 'velocitiesTurbines original ', velocitiesTurbines
    # find turbine powers
    wt_power = np.power(velocitiesTurbines, 3.0) * (0.5*rho*rotorArea*Cp*np.power(np.cos(yaw), pP))
    wt_power /= 1000  # in kW
    power = np.sum(wt_power)

    return velocitiesTurbines, wt_power, power, ws_array


class GaussianWake(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(GaussianWake, self).__init__()

        self.fd_options['form'] = 'central'
        self.fd_options['step_size'] = 1.0e-6
        self.fd_options['step_type'] = 'relative'
        self.fd_options['force_fd'] = True

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        if options is None:
            self.radius_multiplier = 1.0
            self.nSamples = nSamples = 0
        else:
            # self.radius_multiplier = options['radius multiplier']
            self.nSamples = nSamples = options['nSamples']

        # unused but required for compatibility
        self.add_param('hubHeight', np.zeros(nTurbines), units='m')
        self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))

        # used
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineZ', val=np.zeros(nTurbines), units='m')
        self.add_param('yaw%i' % direction_id, np.zeros(nTurbines), units='deg')
        self.add_param('rotorDiameter', val=np.zeros(nTurbines)+126.4, units='m')
        self.add_param('Ct', np.zeros(nTurbines), desc='Turbine thrust coefficients')
        self.add_param('wind_speed', val=8.0, units='m/s')
        self.add_param('axialInduction', val=np.zeros(nTurbines)+1./3.)

        self.add_param('model_params:ke', val=0.052)
        self.add_param('model_params:rotation_offset_angle', val=1.87, units='deg')
        self.add_param('model_params:spread_angle', val=6.37, units='deg')

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

        ke = params['model_params:ke']
        rotation_offset_angle = params['model_params:rotation_offset_angle']
        spread_angle = params['model_params:spread_angle']

        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        yaw = params['yaw%i' % direction_id]
        rotorDiameter = params['rotorDiameter']
        Ct = params['Ct']
        axialInduction = params['axialInduction']
        wind_speed = params['wind_speed']

        yaw *= np.pi/180.
        rotation_offset_angle *= np.pi/180.
        spread_angle *= np.pi/180.0

        if self.nSamples > 0:
            velX = params['wsPositionXw']
            velY = params['wsPositionYw']
            velZ = params['wsPositionZ']
        else:
            velX = np.zeros([0, 0])
            velY = np.zeros([0, 0])
            velZ = np.zeros([0, 0])

        # calculate crosswind locations of wake centers at downstream locations of interest
        wakeCentersY = np.zeros((nSamples, nTurbines))
        wakeCentersYT = np.zeros((nTurbines, nTurbines))
        for turb in range(0, nTurbines):
            wakeAngleInit = 0.5*np.sin(yaw[turb])*Ct[turb] + rotation_offset_angle
            # print turb
            for loc in range(0, nSamples):  # at velX-locations
                deltax = velX[loc]-turbineXw[turb]
                wakeCentersY[loc, turb] = turbineYw[turb]
                if deltax > 0.0:

                    # calculate distance from wake cone apex to wake producing turbine
                    x1 = 0.5*rotorDiameter[turb]/np.tan(spread_angle)

                    # calculate x position with cone apex as origin
                    x = x1 + deltax

                    # calculate wake offset due to yaw
                    wakeCentersYT[loc, turb] -= -wakeAngleInit*(x1**2)/x + x1*wakeAngleInit

            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                wakeCentersYT[turbI, turb] = turbineYw[turb]

                if deltax > 0.0:

                    # calculate distance from wake cone apex to wake producing turbine
                    x1 = 0.5*rotorDiameter[turb]/np.tan(spread_angle)

                    # calculate x position with cone apex as origin
                    x = x1 + deltax

                    # calculate wake offset due to yaw
                    wakeCentersYT[turbI, turb] -= -wakeAngleInit*(x1**2)/x + x1*wakeAngleInit

        # calculate wake zone diameters at locations of interest
        wakeDiameters = np.zeros((nSamples, nTurbines))
        wakeDiametersT = np.zeros((nTurbines, nTurbines))
        for turb in range(0, nTurbines):
            for loc in range(0, nSamples):  # at velX-locations
                deltax = velX[loc]-turbineXw[turb]
                if deltax > 0.0:
                    wakeDiameters[loc, turb] = rotorDiameter[turb]+2.*ke*deltax
            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                if deltax > 0.0:
                    wakeDiametersT[turbI, turb] = rotorDiameter[turb]+2.0*np.tan(spread_angle)*deltax
                    # wakeDiametersT[turbI, turb] = rotorDiameter[turb]+2.0*ke*deltax

        velocitiesTurbines = np.tile(wind_speed, nTurbines)
        # print velocitiesTurbines
        ws_array = np.tile(wind_speed, nSamples)

        for loc in range(0, nSamples):
            wakeEffCoeff = 0
            for turb in range(0, nTurbines):
                deltax = velX[loc] - turbineXw[turb]
                radiusLoc = abs(velY[loc]-wakeCentersY[loc, turb])

                if deltax > 0 and radiusLoc < wakeDiameters[loc, turb]/2.0:
                    sigma = wakeDiametersT[loc, turb]/6.
                    mu = wakeCentersY[loc, turb]
                    max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2.0*ke*deltax), 2.0)

                    wakeEffCoeffTurbine = GaussianMax(velY[loc], max, mu, sigma)
                    wakeEffCoeff += np.power(axialInduction[turb]*wakeEffCoeffTurbine, 2.0)

            wakeEffCoeff[loc] = (1. - 2. * np.sqrt(wakeEffCoeff))
            ws_array[loc] *= wakeEffCoeff

        for turbI in range(0, nTurbines):

            wakeEffCoeff = 0.
            for turb in range(0, nTurbines):

                deltax = turbineXw[turbI] - turbineXw[turb]

                if deltax > 0.:
                    sigma = wakeDiametersT[turbI, turb]/6.
                    mu = wakeCentersYT[turbI, turb]
                    max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2.0*ke*deltax), 2.0)

                    wakeEffCoeffTurbine = GaussianMax(turbineYw[turbI], max, mu, sigma)
                    wakeEffCoeff += np.power(axialInduction[turb]*wakeEffCoeffTurbine, 2.0)

            wakeEffCoeff = (1. - 2. * np.sqrt(wakeEffCoeff))

            velocitiesTurbines[turbI] *= wakeEffCoeff

        unknowns['wtVelocity%i' % direction_id] = velocitiesTurbines
        # print velocitiesTurbines
        if nSamples > 0.0:
            unknowns['wsArray%i' % direction_id] = ws_array


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

    prob['model_params:spread_angle'] = 7.0
    prob['model_params:ke'] = 0.052

    prob['turbineXw'] = turbineX
    prob['turbineYw'] = turbineY

    GaussianWakeVelocity = list()

    yawrange = np.linspace(-40., 40., 400)

    for yaw1 in yawrange:

        prob['yaw0'] = np.array([yaw1, 0.0])
        prob['Ct'] = Ct*np.cos(prob['yaw0']*np.pi/180.)**2
        print prob['Ct']

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
    # yaw = np.array([0.0, 0.0])
    # wind_direction = 0.0
    # FLORISvelocity = list()
    # # FLORISvelocityStat = list()
    # FLORISvelocityLES = list()
    # for pos2 in posrange:
    #
    #     turbineX = np.array([0, 7.*rotorDiameter[0]])
    #     turbineY = np.array([0, pos2])
    #     # print turbineX, turbineY
    #
    #     # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
    #     #                                                                                 air_density, rotorDiameter,
    #     #                                                                                 yaw, Cp, axialInduction,
    #     #                                                                                 turbineX, turbineY)
    #
    #     # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
    #     #                                                                                 air_density, rotorDiameter,
    #     #                                                                                 yaw, Cp, axialInduction,
    #     #                                                                                 turbineX, turbineY)
    #
    #     velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #
    #
    #
    #     velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
    #                                                            yaw, Cp, axialInduction, turbineX, turbineY,)
    #
    #     # print 'power = ', myFloris.root.dir0.unknowns['wt_power']
    #
    #     FLORISvelocity.append(list(velocitiesTurbines))
    #     # FLORISvelocityStat.append(list(velocitiesTurbines_stat))
    #     # FLORISvelocityLES.append(list(velocitiesTurbines_LES))
    #     FLORISvelocityGAZ.append(list(velocitiesTurbines_GAZ))
    #     FLORISvelocityGAR.append(list(velocitiesTurbines_GAR))
    #     FLORISvelocityGARZ.append(list(velocitiesTurbines_GARZ))
    #     FLORISvelocityGH.append(list(velocitiesTurbines_GH))
    #     FLORISvelocityCos.append(list(velocitiesTurbines_Cos))
    #
    #
    # FLORISvelocity = np.array(FLORISvelocity)
    # # FLORISvelocityStat = np.array(FLORISvelocityStat)
    # # FLORISvelocityLES = np.array(FLORISvelocityLES)
    # FLORISvelocityGAZ = np.array(FLORISvelocityGAZ)
    # FLORISvelocityGAR = np.array(FLORISvelocityGAR)
    # FLORISvelocityGARZ = np.array(FLORISvelocityGARZ)
    # FLORISvelocityGH = np.array(FLORISvelocityGH)
    # FLORISvelocityCos = np.array(FLORISvelocityCos)
    #
    #
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocity[:, 1], 'b', label='Floris Original')
    # # axes[2].plot(posrange, FLORISvelocityStat[:, 1], 'g', label='added gaussian')
    # # axes[2].plot(posrange/rotorDiameter[0], FLORISvelocityLES[:, 1], 'c', label='Linear Exact Sector')
    # # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGAZ[:, 1], 'm', label='GAZ')
    # # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGAR[:, 1], 'c', label='GAR')
    # # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGARZ[:, 1], 'g', label='GARZ')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGH[:, 1], 'y', label='GH')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityCos[:, 1], 'r', label='Cos')
    #
    # axes[1, 0].set_xlabel('y/D')
    # axes[1, 0].set_ylabel('Velocity (m/s)')
    # # plt.legend()
    # # plt.show()
    #
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