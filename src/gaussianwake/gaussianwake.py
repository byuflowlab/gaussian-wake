import numpy as np
from scipy.integrate import quad
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

        self.add_param('model_params:ke', val=0.052, pass_by_object=True)
        self.add_param('model_params:rotation_offset_angle', val=1.56, units='deg', pass_by_object=True)
        self.add_param('model_params:spread_angle', val=5.84, units='deg', pass_by_object=True)
        self.add_param('model_params:ky', val=0.5, pass_by_object=True)
        self.add_param('model_params:Dw0', val=np.ones(3)*1.4, pass_by_object=True)
        self.add_param('model_params:m', val=np.ones(3)*0.33, pass_by_object=True)
        self.add_param('model_params:n_std_dev', val=4, desc='the number of standard deviations from the mean that are '
                                                             'assumed included in the wake diameter',
                       pass_by_object=True)
        self.add_param('model_params:integrate', val=True, desc='if True, will integrate over the full rotor for Ueff',
                       pass_by_object=True)
        self.add_param('model_params:extra_diams', val=2,
                       desc='how many diameters the wake width should be at the rotor', pass_by_object=True)
        self.add_param('model_params:spread_mode', 'linear',
                       desc='how the wake expands: linear or power law', pass_by_object=True)
        self.add_param('model_params:yaw_mode', 'linear',
                       desc='how the wake expands: linear or power law', pass_by_object=True)
        self.add_param('model_params:yshift', 0.0,
                       desc='constant offset in crosswind wake position', pass_by_object=True)

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
        n_std_dev = params['model_params:n_std_dev']
        ky = params['model_params:ky']
        Dw0 = params['model_params:Dw0']
        m = params['model_params:m']
        integrate = params['model_params:integrate']
        spread_mode = params['model_params:spread_mode']
        yaw_mode = params['model_params:yaw_mode']
        yshift = params['model_params:yshift']

        # print Dw0

        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['turbineZ']
        yaw = params['yaw%i' % direction_id]
        rotorDiameter = params['rotorDiameter']*2
        Ct = params['Ct']
        axialInduction = params['axialInduction']
        wind_speed = params['wind_speed']

        # Dw0[0] = Dw0[1]
        # m[0] = m[1]

        # rotorDiameter *= 2

        for i in range(0, nTurbines):
            if (Ct[i] > 0.96): # Glauert condition
                axialInduction[i] = 0.143 + np.sqrt(0.0203-0.6427*(0.889 - Ct[i]))
            else:
                axialInduction[i] = 0.5*(1.0-np.sqrt(1.0-Ct[i]))

        # print axialInduction
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
            # wakeAngleInit = 0.5*np.sin(yaw[turb])*Ct[turb] + rotation_offset_angle
            # print turb
            for loc in range(0, nSamples):  # at velX-locations
                deltax = velX[loc]-turbineXw[turb]
                wakeCentersY[loc, turb] = turbineYw[turb]
                if deltax > 0.0:

                    wakeCentersY[loc, turb] += get_wake_offset(deltax, yaw[turb], rotorDiameter[turb], Ct[turb],
                                                               rotation_offset_angle, mode=yaw_mode, ky=ky,
                                                               Dw0=Dw0[0], m=m[0], yshift=yshift)

            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                wakeCentersYT[turbI, turb] = turbineYw[turb]

                if deltax > 0.0:

                    wakeCentersYT[turbI, turb] += get_wake_offset(deltax, yaw[turb], rotorDiameter[turb], Ct[turb],
                                                                  rotation_offset_angle, mode=yaw_mode, ky=ky,
                                                                  Dw0=Dw0[0], m=m[0], yshift=yshift)

        # calculate wake zone diameters at locations of interest
        wakeDiameters = np.zeros((nSamples, nTurbines))
        wakeDiametersT = np.zeros((nTurbines, nTurbines))
        for turb in range(0, nTurbines):
            for loc in range(0, nSamples):  # at velX-locations
                deltax = velX[loc]-turbineXw[turb]
                if deltax > 0.0:
                    wakeDiameters[loc, turb] = get_wake_diameter(deltax, rotorDiameter[turb], spread_mode, spread_angle,
                                                                 Dw0=Dw0[1], m=m[1], ke=ke)
                    # wakeDiameters[loc, turb] = rotorDiameter[turb]+2.0*np.tan(spread_angle)*deltax
                    # if wakeDiameters[loc, turb] < 126.4:
                    # print wakeDiameters[loc, turb]
                    # quit()
            for turbI in range(0, nTurbines):  # at turbineX-locations
                deltax = turbineXw[turbI]-turbineXw[turb]
                if deltax > 0.0:
                    wakeDiametersT[turbI, turb] = get_wake_diameter(deltax, rotorDiameter[turb], spread_mode, spread_angle,
                                                                    Dw0=Dw0[1], m=m[1], ke=ke, Ct=Ct[turb])
                    # wakeDiametersT[turbI, turb] = np.cos(yaw[turb])*rotorDiameter[turb]+2.0*np.tan(spread_angle)*deltax
                    # wakeDiametersT[turbI, turb] = rotorDiameter[turb]+2.0*np.tan(spread_angle)*deltax
                    # wakeDiametersT[turbI, turb] = rotorDiameter[turb]+2.0*ke*deltax

        velocitiesTurbines = np.tile(wind_speed, nTurbines)
        # print velocitiesTurbines
        ws_array = np.tile(wind_speed, nSamples)

        for loc in range(0, nSamples):
            wakeEffCoeff = 0
            for turb in range(0, nTurbines):
                deltax = velX[loc] - turbineXw[turb]

                if deltax > 0:
                    R = abs(wakeCentersY[loc, turb] - velY[loc])
                    wakeEffCoeffTurbine = get_wake_deficit_point(R, deltax, wakeDiameters[loc, turb],
                                                                 rotorDiameter[turb], axialInduction[turb],
                                                                 ke, n_std_dev)

                    wakeEffCoeff += np.power(wakeEffCoeffTurbine, 2.0)

            wakeEffCoeff = (1. - np.sqrt(wakeEffCoeff))
            ws_array[loc] *= wakeEffCoeff

        if integrate:
            for turbI in range(0, nTurbines):
                wakeEffCoeff = 0.
                for turb in range(0, nTurbines):

                    deltax = turbineXw[turbI] - turbineXw[turb]

                    if deltax > 0.:
                        deltay = abs(wakeCentersYT[turbI, turb] - turbineYw[turbI])
                        a = deltay - 0.5*rotorDiameter[turbI]
                        b = deltay + 0.5*rotorDiameter[turbI]
                        wakeEffCoeffTurbine, _ = quad(get_wake_deficit_integral, a, b,
                                                   (deltax, deltay, wakeDiametersT[turbI, turb], rotorDiameter[turbI],
                                                    axialInduction[turb], ke, n_std_dev))
                        wakeEffCoeffTurbine /= 0.25*np.pi*rotorDiameter[turbI]**2
                        wakeEffCoeff += np.power(wakeEffCoeffTurbine, 2.0)

                wakeEffCoeff = (1. - np.sqrt(wakeEffCoeff))

                velocitiesTurbines[turbI] *= wakeEffCoeff

        else:
            if spread_mode is 'bastankhah':
                indices = sorted(range(len(turbineXw)), key=lambda kidx: -turbineXw[kidx])
                for turbI in range(0, nTurbines):
                    wakeEffCoeff = 0.

                    for indx in indices:

                        deltax = turbineXw[turbI] - turbineXw[indx]

                        if deltax > 0.:

                            R = abs(wakeCentersYT[turbI, indx] - turbineYw[turbI])
                            wakeEffCoeffTurbine = get_wake_deficit_point(R, deltax, wakeDiametersT[turbI, indx],
                                                                         rotorDiameter[indx], axialInduction[indx], ke,
                                                                         Ct[indx], yaw[indx], n_std_dev, Dw0[2], m[2],
                                                                         mode=spread_mode)

                            wakeEffCoeff += velocitiesTurbines[indx]*wakeEffCoeffTurbine
                            #
                            # half_width = (0.5/velocitiesTurbines[turbI])*get_wake_deficit_point(0.0, deltax, wakeDiametersT[turbI, indx],
                            #                                                                     rotorDiameter[indx], axialInduction[indx], ke,
                            #                                                                     Ct[indx], yaw[indx], n_std_dev, Dw0[2], m[2],
                            #                                                                     mode=spread_mode)
                            # print 'hw', half_width
                    velocitiesTurbines[turbI] -= wakeEffCoeff
                # print velocitiesTurbines

            else:
                for turbI in range(0, nTurbines):
                        for turb in range(0, nTurbines):
                            wakeEffCoeff = 0.

                            deltax = turbineXw[turbI] - turbineXw[turb]

                            if deltax > 0.:

                                R = abs(wakeCentersYT[turbI, turb] - turbineYw[turbI])
                                wakeEffCoeffTurbine = get_wake_deficit_point(R, deltax, wakeDiametersT[turbI, turb],
                                                                             rotorDiameter[turbI], axialInduction[turb], ke,
                                                                             Ct[turb], yaw[turb], n_std_dev, Dw0[2], m[2],
                                                                             mode=spread_mode)

                                wakeEffCoeff += np.power(wakeEffCoeffTurbine, 2.0)

                        wakeEffCoeff = (1. - np.sqrt(wakeEffCoeff))

                        velocitiesTurbines[turbI] *= wakeEffCoeff

        unknowns['wtVelocity%i' % direction_id] = velocitiesTurbines
        # print velocitiesTurbines
        if nSamples > 0.0:
            print nSamples
            unknowns['wsArray%i' % direction_id] = ws_array
            print unknowns['wsArray%i' % direction_id]


def get_wake_offset(deltax, yaw, rotor_diameter, Ct, rotation_offset_angle, mode='linear', ky=0.1, Dw0=1.3, m=0.33, yshift=0.0):
    """
    Calculates the wake offset due using the theory of Jimenez 2010
    :param deltax: downstream distance from turbine to POI
    :param rotor_diameter: diameter of the turbine rotor
    :param rotation_offset_angle: added initial wake angle, possibly due to rotation
    :param mode: 'linear' assumes linear wake expansion, 'power' assumes power law wake expansion
    :param ky: parameter controlling spread angle of wake (ky=tan(spreading angle))
    :param Dw0: diameter of wake at 1D downstream in rotor diameters (used for power law)
    :param m: exponent for power law
    :return: wake offset based on yaw
    """
    # deltax += rotor_diameter
    wakeAngleInit = 0.5*np.sin(yaw)*Ct + rotation_offset_angle
    if mode is 'linear':

        # calculate distance from wake cone apex to wake producing turbine
        x1 = 0.5*rotor_diameter/ky

        # calculate x position with cone apex as origin
        x = x1 + deltax

        # calculate wake offset due to yaw (see Jimenez et. al 2010)
        wake_center_offset = -wakeAngleInit*(x1**2)/x + x1*wakeAngleInit

    elif mode is 'power':

        # change Dw0 to meters
        Dw0 *= rotor_diameter

        # # calculate the wake offset based on a power law (see Jimenez et. al 2010 and Aitken et. al 2014)
        # wake_center_offset = -wakeAngleInit*pow(rotor_diameter, 2.*(m+1.))*pow(Dw0, -2)*(deltax/((2*m-1)*pow(deltax, 2.*m)))

        # corrected (re-derived) power law yaw model (see Jimenez et. al 2010 and Aitken et. al 2014
        wake_center_offset = -wakeAngleInit*(np.power(deltax, -2.*m+1.)*np.power(rotor_diameter, 2.+2.*m))/((2.*m-1.)*Dw0**2)

    elif mode is 'bastankhah':

        beta = 0.5*((1.+np.sqrt(1.-Ct))/np.sqrt(1.-Ct))
        epsilon = 0.2*np.sqrt(beta)

        # deltay = (0.25)*xi*rotor_diameter*x/(epsilon*(rotor_diameter*epsilon+k*x))
        wake_center_offset = (0.25)*np.power(rotor_diameter, 2)*wakeAngleInit*(deltax-2.)/((rotor_diameter*epsilon+2*ky)*(rotor_diameter*epsilon+ky*deltax))

    else:
        raise KeyError('Invalid wake offset calculation mode')
    wake_center_offset += yshift
    return -wake_center_offset


def get_wake_diameter(deltax, rotor_diameter, mode='linear', spread_angle=7.0, Dw0=1.3, m=0.33, ke=0.075, Ct=0.8):
    """
    Calculates the diameter of the turbine wake
    :param deltax: downstream distance from hub to point of interest in meters
    :param rotor_diameter: diameter of the turbine rotor in meters
    :param mode: 'linear' assumes linear wake expansion, 'power' assumes power law wake expansion
    :param spread_angle: spreading angle of the turbine wake in deg.
    :param Dw0: diameter of wake at 1D downstream in rotor diameters (used for power law)
    :param m: exponent used for power law
    :return: wake diameter at downstream distance of interest
    """
    # deltax += rotor_diameter
    if mode is 'linear':
        wake_diameter = rotor_diameter+2.0*np.tan(spread_angle)*deltax

    elif mode is 'power':
        Dw0 *= rotor_diameter
        wake_diameter = Dw0*(deltax/rotor_diameter)**m
        # wake_diameter = rotor_diameter + Dw0*(deltax/rotor_diameter)**m

        # print 'wake_diameter: ', wake_diameter
    elif mode is 'bastankhah':
        beta = 0.5*((1.+np.sqrt(1.-Ct))/np.sqrt(1.-Ct))
        epsilon = 0.2*np.sqrt(beta)
        wake_diameter = 2.*rotor_diameter*(epsilon + ke*deltax/rotor_diameter)
    else:
        raise KeyError('Invalid wake diameter calculation mode')

    return wake_diameter


def get_wake_deficit_point(R, deltax, wake_diameter, rotor_diameter, axial_induction, ke, Ct, yaw, n_std_dev=4,
                           Dw0=1.3, m=0.33, mode='linear'):
    """
    Calculate the velocity deficit at a point in the wind turbine wake
    :param R: distance from the wake center to the point of interest
    :param deltax: downwind distance from upstream turbine to the downstream turbine
    :param wake_diameter: diameter of the wake of the upstream turbine at the downstream turbine
    :param rotor_diameter: diameter of the downstream turbine rotor
    :param n_std_dev: how many standard deviations to include in the wake diameter (two sided)
    :param ke: entrainment constant (see Jensen 1983)
    :return: velocity deficit at the point of interest
    """
    # deltax += rotor_diameter
    Dw0 *= rotor_diameter

    sigma = wake_diameter/n_std_dev
    mu = 0.0

    if mode is 'bastankhah':

        beta = 0.5*((1.+np.sqrt(1.-Ct))/np.sqrt(1.-Ct))
        epsilon = 0.2*np.sqrt(beta)
        Rwn = epsilon + ke*deltax/rotor_diameter
        tmp0 = Rwn**2
        deficit = (1.-np.sqrt(1.-(Ct/(8.*tmp0))))\
                  *np.exp((-1./(2.*tmp0))*((2.*R/rotor_diameter)**2))
    else:
        # linear
        if mode is 'linear':
            max = 2.*axial_induction*np.power((rotor_diameter)/(rotor_diameter+2.0*ke*deltax), 2.0)
        # power
        elif mode is 'power':
            # max = (axial_induction-1.)*np.power(rotor_diameter/(Dw0*(deltax/rotor_diameter)**m), 2.0)
            # power with 2a
            # max = (2.*axial_induction)*np.power(rotor_diameter/(Dw0*(deltax/rotor_diameter)**m), 2.0)
            # power with 2a and offset
            # max = (2.*axial_induction)*np.power(rotor_diameter/(rotor_diameter + Dw0*(deltax/rotor_diameter)**m), 2.0)
            # power per Aitken et al. 2014
            Dw0 /= rotor_diameter
            max = Dw0*2.*axial_induction*(deltax/rotor_diameter)**m
            # max = 1.5*axial_induction*(deltax/rotor_diameter)**m

            # re-derived deficit using momentum balance and actuator disk theory as per Jimenez, with a power law wake
            # expansion
            # max = 0.5 + 0.5*np.sqrt(1.-2.*Ct*np.sin(yaw)*(rotor_diameter/wake_diameter)**2)
            # max = 0.5*np.sqrt(1.-2.*Ct*np.sin(yaw)*(rotor_diameter/wake_diameter)**2)

        deficit = GaussianMax(R, max, mu, sigma)

    return deficit


def get_wake_deficit_integral(R, deltax, deltay, wake_diameter, rotor_diameter, axial_induction, ke,
                              n_std_dev=4, Dw0=1.3, m=0.33):
    """
    Return the integrand for wake deficit using exact circular sections
    :param R: distance from the wake center to the point of interest
    :param deltax: downwind distance from upstream turbine to the downstream turbine
    :param deltay: crosswind distance from the center to downstream turbine hub
    :param wake_diameter: diameter of the wake of the upstream turbine at the downstream turbine
    :param rotor_diameter: diameter of the downstream turbine rotor
    :param n_std_dev: how many standard deviations to include in the wake diameter (two sided)
    :param ke: entrainment constant (see Jensen 1983)
    :return: integrand for integrating the velocity deficit over the rotor-swept area
    """

    # calculate the rotor radius of the downstream turbine
    rotor_radius = 0.5*rotor_diameter

    # if radius is very small, just set deficit to zero
    if R < 1E-12:
        # print "1"
        integration_angle = 0.0

    # if rotor overlaps the wake center then we have to account for full circular section of the wake
    elif (deltay < rotor_radius) and (R < abs(rotor_radius-deltay)):
        # print "2"
        integration_angle = 2.0*np.pi

    # if rotor does not overlap the wake center, then use the arc angle of two overlapping circles
    else:
        # print "3"
        integration_angle = 2.0*np.arccos((deltay**2+R**2-rotor_radius**2)/(2.*deltay*R))

    # get the deficit at the relevant radial location
    deficit = get_wake_deficit_point(R, deltax, wake_diameter, rotor_diameter, axial_induction, ke, n_std_dev, Dw0, m)

    # return the thing we are integrating
    return deficit*integration_angle*R

# def get_point_deficit(R, yaw, Ct, rotation_offset_angle, turbineYw, rotor_diameter, ky, deltax, spread_angle, n_std_dev, ke):
#     # ############# Calculate the wake center #################
#     wakeAngleInit = 0.5*np.sin(yaw)*Ct + rotation_offset_angle
#
#     wakeCenterYT = turbineYw
#
#     # calculate distance from wake cone apex to wake producing turbine
#     x1 = 0.5*rotorDiameter/ky
#
#     # calculate x position with cone apex as origin
#     x = x1 + deltax
#
#     # calculate wake offset due to yaw
#     wakeCenterYT -= -wakeAngleInit*(x1**2)/x + x1*wakeAngleInit
#     # ##########################################################
#
#     # ############ Calculate wake diameter ######################
#
#     wakeDiameterT = wake_diameter_linear(deltax, rotor_diameter, spread_angle)
#     # wakeDiameterT = wake_diameter_power(deltax, rotor_diameter)
#
#     # ##########################################################
#
#     sigma = wakeDiameterT/n_std_dev
#     mu = wakeCenterYT
#
#     max = 2.*axialInduction*np.power((rotor_diameter)/(rotorDiameter+2.0*ke*deltax), 2.0)
#
#     deficit = GaussianMax(R, max, mu, sigma)
#
#     return deficit

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