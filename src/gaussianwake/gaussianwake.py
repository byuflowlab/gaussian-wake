import numpy as np
import scipy as sp
from scipy.io import loadmat
import pylab as plt
import time

from openmdao.api import Component


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
                mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb])
                sigma = (wakeDiametersT[turbI, turb] + rotorDiameter[turbI])/6.
                mu = wakeCentersYT[turbI, turb]
                max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)
                # p1 = wakeCentersYT[turbI, turb] + deltay - rotorDiameter[turbI]/2.
                # p2 = wakeCentersYT[turbI, turb] + deltay + rotorDiameter[turbI]/2.

                # wakeEffCoeffPerZone = GaussianAverage(max, mu, sigma, p1, p2)
                wakeEffCoeffPerZone = GaussianMax(turbineY[turbI], max, mu, sigma)
                wakeEffCoeff = wakeEffCoeff + np.power(axialInd[turb]*wakeEffCoeffPerZone, 2.0)


        wakeEffCoeff = (1 - 2 * np.sqrt(wakeEffCoeff))
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
            nSamples = 0
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

        self.add_param('model_params:ke', val=0.065)
        self.add_param('model_params:kd', val=0.15)
        self.add_param('model_params:me', val=0.72508)
        self.add_param('model_params:MU', val=0.80415)
        self.add_param('model_params:aU', val=5.0)
        self.add_param('model_params:bU', val=1.66)

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
        kd = params['model_params:kd']
        me = params['model_params:me']
        MU = params['model_params:MU']
        aU = params['model_params:aU']
        bU = params['model_params:bU']

        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        yaw = params['yaw%i' % direction_id]
        rotorDiameter = params['rotorDiameter']
        Ct = params['Ct']
        axialInduction = params['axialInduction']
        wind_speed = params['wind_speed']

        yaw = yaw*np.pi/180.

        if self.nSamples > 0:
            velX = params['wsPositionXw']
            velY = params['wsPositionYw']
            velZ = params['wsPositionZ']
        else:
            velX = np.zeros([0, 0])
            velY = np.zeros([0, 0])
            velZ = np.zeros([0, 0])

        # calculate y-location of wake centers
        wakeCentersY = np.zeros((nSamples, nTurbines))
        wakeCentersYT = np.zeros((nTurbines, nTurbines))
        for turb in range(0, nTurbines):
            # wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]), 2)*np.sin(yaw[turb])*4*axialInduction[turb]*(1-axialInduction[turb])
            wakeAngleInit = 0.5*np.sin(yaw[turb])*Ct[turb]
            for loc in range(0, nSamples):  # at velX-locations
                deltax = np.maximum(velX[loc]-turbineXw[turb], 0)
                factor = (2*kd*deltax/rotorDiameter[turb])+1
                wakeCentersY[loc, turb] = turbineYw[turb]
                # wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + ad+bd*deltax  # rotation-induced deflection
                wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + \
                    (wakeAngleInit*(15.*np.power(factor, 4)+np.power(wakeAngleInit, 2))/((30.*kd*np.power(factor, 5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15.+np.power(wakeAngleInit, 4))/(30.*kd))  # yaw-induced deflection
            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                if deltax > 0:
                    factor = (2*kd*deltax/rotorDiameter[turb])+1
                    wakeCentersYT[turbI, turb] = turbineYw[turb]
                    # wakeCentersYT[turbI, turb] = wakeCentersYT[turbI, turb] + ad+bd*deltax  # rotation-induced deflection
                    wakeCentersYT[turbI, turb] = wakeCentersYT[turbI, turb] + \
                        (wakeAngleInit*(15.*np.power(factor, 4)+np.power(wakeAngleInit, 4))/((30.*kd*np.power(factor, 5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15.+np.power(wakeAngleInit, 4))/(30.*kd))  # yaw-induced deflection

        # calculate wake zone diameters at velX-locations
        wakeDiameters = np.zeros((nSamples, nTurbines))
        wakeDiametersT = np.zeros((nTurbines, nTurbines))
        for turb in range(0, nTurbines):
            for loc in range(0, nSamples):  # at velX-locations
                deltax = velX[loc]-turbineXw[turb]
                wakeDiameters[loc, turb] = rotorDiameter[turb]+2.*ke*me*np.maximum(deltax, 0)
            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                wakeDiametersT[turbI, turb] = np.maximum(rotorDiameter[turb]+2.*ke*me*deltax, 0)

        ke = np.ones(nTurbines)*ke

        # calculate velocities in full flow field (optional)
        ws_array = np.tile(wind_speed, nSamples)
        for turb in range(0, nTurbines):
            mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
            for loc in range(0, nSamples):
                deltax = velX[loc] - turbineXw[turb]
                radiusLoc = abs(velY[loc]-wakeCentersY[loc, turb])
                axialIndAndNearRotor = 2*axialInduction[turb]

                if deltax > 0 and radiusLoc < wakeDiameters[loc, turb]/2.0:    # check if in wake
                    reductionFactor = axialIndAndNearRotor * \
                        np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2.*ke[turb]*(mU)*deltax)), 2)
                else:
                    reductionFactor = 0
                ws_array[loc] *= (1.-reductionFactor)

        # find effective wind speeds at downstream turbines, then predict power downstream turbine
        velocitiesTurbines = np.tile(wind_speed, nTurbines)

        for turbI in range(0, nTurbines):

            # find overlap-area weighted effect of each wake zone
            wakeEffCoeff = 0
            for turb in range(0, nTurbines):

                deltax = turbineXw[turbI] - turbineXw[turb]

                if deltax > 0:
                    # deltay = wakeCentersYT[turbI, turb] - turbineY[turbI]
                    mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb])
                    sigma = (wakeDiametersT[turbI, turb] + rotorDiameter[turbI])/6.
                    mu = wakeCentersYT[turbI, turb]
                    max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)

                    wakeEffCoeffTurbine = GaussianMax(turbineYw[turbI], max, mu, sigma)
                    wakeEffCoeff += np.power(axialInduction[turb]*wakeEffCoeffTurbine, 2.0)

            wakeEffCoeff = (1. - 2. * np.sqrt(wakeEffCoeff))

            # multiply the inflow speed with the wake coefficients to find effective wind speed at turbine
            velocitiesTurbines[turbI] *= wakeEffCoeff

        unknowns['wtVelocity%i' % direction_id] = velocitiesTurbines