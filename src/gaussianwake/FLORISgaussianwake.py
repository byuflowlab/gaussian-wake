import numpy as np
import scipy as sp
from scipy.io import loadmat
import pylab as plt
import time


def FLORIS_Stat(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""


    pP = 1.88
    ke = 0.065
    keCorrDA = 0.0
    kd = 0.15
    me = np.array([-0.5,0.22,1.0])
    MU = np.array([0.5,1.0,5.5])
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180.        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = (np.pi*rotorDiameter**2.)/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction

    yaw = yaw*np.pi/180.

    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix,np.array([turbineX,turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0,nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines,nTurbines,3))
    for turb in range(0,nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(0,3):
                wakeDiametersT[turbI,turb,zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax,0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    wakeOverlapTRel = wakeOverlapT
    for turb in range(0,nTurbines):
        wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    keUncorrected = ke
    ke = np.zeros(nTurbines)
    for turb in range(0,nTurbines):
        s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
        ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0,deltax))),2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0,deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf,nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0,nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]
            deltay = turbineY[turbI] - wakeCentersYT[turbI, turb]
            # if (turb != turbI):
            # print 'y, yw: ', turbineY[turb], wakeCentersYT[turbI, turb]
            if deltax > 0:
                mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb])

                for zone in range(0,3):
                    N = gaussian(turbineY[turbI], wakeCentersYT[turbI, turb], wakeDiametersT[turbI, turb, 2]/4.)
                    # print 'N = ', N
                    wakeEffCoeffPerZone = wakeEffCoeffPerZone + (np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU[zone]*deltax) + N, 2.0) + 0.0*N) * wakeOverlapTRel[turbI,turb,zone]

                wakeEffCoeff = wakeEffCoeff + np.power(axialInd[turb]*wakeEffCoeffPerZone, 2.0)

        wakeEffCoeff = (1 - 2 * np.sqrt(wakeEffCoeff))

        # multiply the inflow speed with the wake coefficients to find effective wind speed at turbine
        velocitiesTurbines[turbI] *= wakeEffCoeff

    # find turbine powers
    # print yaw, pP
    wt_power = np.power(velocitiesTurbines, 3.0) * (0.5*rho*rotorArea*Cp*np.power(np.cos(yaw), pP))
    wt_power /= 1000  # in kW
    power = np.sum(wt_power)

    return velocitiesTurbines, wt_power, power, ws_array


def FLORIS(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.493681, 0.196292, 0.587587, 0.442133, 1.110093, 5.537140])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""


    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    # me = np.array([-0.5, 0.22,1.0])
    # MU = np.array([0.5, 1.0, 5.5])
    me = xdict['xvars'][0:3]
    MU = xdict['xvars'][3:6]
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction
    #
    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]


    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix,np.array([turbineX,turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0, nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines,nTurbines,3))
    for turb in range(0,nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(0,3):
                wakeDiametersT[turbI,turb,zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax,0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    wakeOverlapTRel = wakeOverlapT
    for turb in range(0,nTurbines):
        wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    keUncorrected = ke
    ke = np.zeros(nTurbines)
    for turb in range(0,nTurbines):
        s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
        ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0,deltax))),2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0,deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf,nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0,nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]

            if deltax > 0:
                mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb])
                for zone in range(0,3):
                    wakeEffCoeffPerZone = wakeEffCoeffPerZone + np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU[zone]*deltax),2.0) * wakeOverlapTRel[turbI,turb,zone]

                    # print "wake effective per zone Original", wakeEffCoeffPerZone
                wakeEffCoeff = wakeEffCoeff + np.power(axialInd[turb]*wakeEffCoeffPerZone,2.0)


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


def FLORIS_LinearizedExactSector(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""


    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    me = np.array([-0.5, 0.22, 1.0, 1.1])
    MU = np.array([0.4, 0.5, 1.0, 5.5, 5.6])*np.array([0.75, 5., 20., 20., 20.])
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction
    #
    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]


    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix,np.array([turbineX,turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0,nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines, nTurbines, 5))
    for turb in range(0, nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0, nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(1, 4):
                wakeDiametersT[turbI, turb, zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax, 0)
            wakeDiametersT[turbI, turb, 0] = 0.0



    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    #wakeOverlapT = calcOverlapAreas(turbineX, turbineY, rotorDiameter, wakeDiametersT[:, :, 1:4], wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    #wakeOverlapTRel = wakeOverlapT
    #for turb in range(0,nTurbines):
    #    wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    #keUncorrected = ke
    #ke = np.zeros(nTurbines)
    #for turb in range(0,nTurbines):
     #   s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
     #   ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    # ws_array = np.tile(Vinf,nLocations)
    # for turb in range(0,nTurbines):
    #     mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
    #     for loc in range(0,nLocations):
    #         deltax = velX[loc] - turbineX[turb]
    #         radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
    #         axialIndAndNearRotor = 2*axialInd[turb]
    #
    #         if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
    #             reductionFactor = axialIndAndNearRotor*\
    #                               np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0, deltax))),2)
    #         elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
    #             reductionFactor = axialIndAndNearRotor*\
    #                               np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0, deltax))),2)
    #         elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
    #             reductionFactor = axialIndAndNearRotor*\
    #                               np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0, deltax))),2)
    #         elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
    #             reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0, deltax)/(rotorDiameter[turb]))/np.pi)
    #         else:
    #             reductionFactor = 0
    #         ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf,nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0.0
        for turb in range(0,nTurbines):
            # if turbI != turb:
            wakeEffCoeffPerZone = 0.0

            deltax = turbineX[turbI] - turbineX[turb]
            deltay = wakeCentersYT[turbI, turb] - turbineY[turbI]

            # if abs(deltay) > rotorDiameter[turbI] + wakeDiametersT[turbI, turb, 4]:
            #     wak

            if deltax > 0.0:

                mU = MU / np.cos(aU*np.pi/180. + bU*yaw[turb])

                C = (rotorDiameter[turb]/(rotorDiameter[turb]+2.*ke*mU*deltax))**2
                if abs(deltay) - rotorDiameter[turbI]/2. <= 0.:
                    R1 = 0.0
                else:
                    R1 = abs(deltay)-rotorDiameter[turbI]/2.
                R2 = abs(deltay)+rotorDiameter[turbI]/2.
                ######## integral = sp.integrate.quad(LinearizedExactSector, R1, R2, args=(rotorDiameter[turbI]/2., deltay, C, wakeDiametersT[turbI, turb]), limit=7)

                ################
                res = 100000
                integral = np.zeros(res)###
                count = 0
                for r in np.linspace(R1, R2, res):
                    integral[count] = LinearizedExactSector(r, rotorDiameter[turbI]/2., deltay, C, wakeDiametersT[turbI, turb])
                    count += 1
                integral = sp.integrate.trapz(integral, np.linspace(R1, R2, res), None)
                #################


                # integral = sp.integrate.quad(circleAreaForInt, 0, 126.4)
                # print 'integral', turbI, turb, R1, R2, integral
                # if (integral[0] > 12549) or (integral[0] < 12547):
                #     raise ValueError('incorrect integral value returned')
                ######## wakeEffCoeffPerZone = integral[0]/(np.pi*(rotorDiameter[turbI]**2)/4.)

                ###################
                wakeEffCoeffPerZone = integral/(np.pi*(rotorDiameter[turbI]**2)/4.)
                ###################

                wakeEffCoeff = wakeEffCoeff + np.power(axialInd[turb]*wakeEffCoeffPerZone, 2.0)

        wakeEffCoeff = (1 - 2 * np.sqrt(wakeEffCoeff))
        # print 'wakeEffCoeff LES', wakeEffCoeff
        # multiply the inflow speed with the wake coefficients to find effective wind speed at turbine
        velocitiesTurbines[turbI] *= wakeEffCoeff
        # print 'velocitiesTurbines LES ', velocitiesTurbines
    # find turbine powers
    wt_power = np.power(velocitiesTurbines, 3.0) * (0.5*rho*rotorArea*Cp*np.power(np.cos(yaw), pP))
    wt_power /= 1000  # in kW
    # print wt_power
    power = np.sum(wt_power)

    return velocitiesTurbines, wt_power, power#, ws_array


def FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.101831, 0.095223, 0.095255, 0.5, 0.510440, 5.500000])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""

    #xdict={'xvars': np.array([-0.101831, 0.095223, 0.095255, 0.5, 0.510440, 5.500000])}
    #xdict={'xvars': np.array([-1.134875, 0.128273, 0.128275, 0.5, 0.556273, 5.500000])}
    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    # me = np.array([-0.5, 0.22, 1.0])*np.array([1., 1., 0.55])
    # MU = np.array([0.5, 1.0, 5.5])*np.array([0., 0.55, 0.0])
    me = xdict['xvars'][0:3]
    MU = xdict['xvars'][3:6]
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction
    #
    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]


    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix,np.array([turbineX,turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0,nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines,nTurbines,3))
    for turb in range(0,nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(0,3):
                wakeDiametersT[turbI,turb,zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax,0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)
    # print wakeOverlapT
    # make overlap relative to rotor area (maximum value should be 1)
    wakeOverlapTRel = np.copy(wakeOverlapT)
    for turb in range(0, nTurbines):
        wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    keUncorrected = ke
    ke = np.zeros(nTurbines)
    for turb in range(0,nTurbines):
        s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
        ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0,deltax))),2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0,deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf,nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0, nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]
            deltay = wakeCentersYT[turbI, turb] - turbineY[turbI]

            if deltax > 0:
                mU = MU[1] / np.cos(aU*np.pi/180 + bU*yaw[turb])
                sigma = (wakeDiametersT[turbI, turb, 2] + rotorDiameter[turbI])/6.
                mu = wakeCentersYT[turbI, turb]
                max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)
                # print wakeDiametersT[turb, turbI, 0]
                # print wakeOverlapT[turbI, turb, 0], (0.25*np.pi*wakeDiametersT[turbI, turb, 0]**2)
                for zone in range(0, 4):
                    p1 = wakeCentersYT[turbI, turb] + deltay - 0.5*rotorDiameter[turbI]
                    p2 = wakeCentersYT[turbI, turb] + deltay + 0.5*rotorDiameter[turbI]
                    if zone == 3:
                        if 1. - sum(wakeOverlapTRel[turbI, turb, :]) >= 1E-8:
                            if wakeOverlapTRel[turbI, turb, zone-1] > 0.:
                                # print 'here here'
                                if wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2) < 1. - 1E-8:
                                    # print 'here'
                                    if deltay > 0.0:
                                        p1 = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone-1]
                                    if deltay < 0.0:
                                        p2 = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone-1]

                            if wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2) >= 1. - 1E-8:
                                # print 'no here'
                                # print wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2)
                                p1p = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone-1]
                                p2p = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone-1]
                                # print p1, p1p, p2p, p2
                                Gave1 = GaussianAverage(max, mu, sigma, p1, p1p)
                                Gave2 = GaussianAverage(max, mu, sigma, p2p, p2)
                                Gave3 = GaussianAverage(max, mu, sigma, p1, p2)
                                # print Gave1, Gave2, Gave3
                                Gave = (Gave1*abs(p1p-p1) + Gave2*abs(p2-p2p))/(abs(p1p-p1)+abs(p2-p2p))
                                # Gave = 0.5*(Gave1 + Gave2)
                                # print Gave
                                # Gave = Gave3
                                wakeEffCoeffPerZone += Gave*(1.-sum(wakeOverlapTRel[turbI, turb, :]))
                            else:
                                wakeEffCoeffPerZone += GaussianAverage(max, mu, sigma, p1, p2)*(1.-sum(wakeOverlapTRel[turbI, turb, :]))

                    elif wakeOverlapTRel[turbI, turb, zone] > 0.0:
                        if wakeCentersYT[turbI, turb] + deltay - 0.5*rotorDiameter[turbI] < wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone]:
                            p1 = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone]
                        if wakeCentersYT[turbI, turb] + deltay + 0.5*rotorDiameter[turbI] > wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone]:
                            p2 = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone]
                        # print zone-1, wakeOverlapTRel[turbI, turb, zone-1], wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2)
                        if (zone > 0.) and (wakeOverlapTRel[turbI, turb, zone-1] > 0.):
                            # print 'here here'
                            if wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2) < 1. - 1E-8:
                                # print 'here'
                                if deltay > 0.0:
                                    p1 = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone-1]
                                if deltay < 0.0:
                                    p2 = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone-1]

                        if zone > 0 and (wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2) >= 1. - 1E-8):
                            # print 'no here'
                            # print wakeOverlapT[turbI, turb, zone-1]/(0.25*np.pi*wakeDiametersT[turbI, turb, zone-1]**2)
                            p1p = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone-1]
                            p2p = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone-1]
                            # print p1, p1p, p2p, p2
                            Gave1 = GaussianAverage(max, mu, sigma, p1, p1p)
                            Gave2 = GaussianAverage(max, mu, sigma, p2p, p2)
                            Gave3 = GaussianAverage(max, mu, sigma, p1, p2)
                            # print Gave1, Gave2, Gave3
                            Gave = (Gave1*abs(p1p-p1) + Gave2*abs(p2-p2p))/(abs(p1p-p1)+abs(p2-p2p))
                            # Gave = 0.5*(Gave1 + Gave2)
                            # print Gave
                            # Gave = Gave3
                            wakeEffCoeffPerZone += Gave*wakeOverlapTRel[turbI, turb, zone]
                        else:
                            wakeEffCoeffPerZone += GaussianAverage(max, mu, sigma, p1, p2)*wakeOverlapTRel[turbI, turb, zone]

                        # print wakeEffCoeffPerZone
                    # print "wake effective per zone Original", wakeEffCoeffPerZone
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


def FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.5, 0.22,  0.0, 0.5, 1.0, 0.531035])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""


    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    # me = np.array([-0.5, 0.22, 1.0])*np.array([0., 0., 0.05])
    # MU = np.array([0.5, 1.0, 5.5])*np.array([0., 0., 0.096])
    me = xdict['xvars'][0:3]
    MU = xdict['xvars'][3:6]
    # print me, MU
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction
    #
    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]


    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix, np.array([turbineX, turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0,nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines,nTurbines,3))
    for turb in range(0,nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(0,3):
                wakeDiametersT[turbI,turb,zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax,0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    wakeOverlapTRel = wakeOverlapT
    for turb in range(0,nTurbines):
        wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    keUncorrected = ke
    ke = np.zeros(nTurbines)
    for turb in range(0,nTurbines):
        s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
        ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0,deltax))),2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0,deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf,nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0, nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]

            if deltax > 0:
                deltay = wakeCentersYT[turbI, turb] - turbineY[turbI]
                mU = MU[2] / np.cos(aU*np.pi/180 + bU*yaw[turb])
                sigma = (wakeDiametersT[turbI, turb, 2] + rotorDiameter[turbI])/6.
                mu = wakeCentersYT[turbI, turb]
                max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)
                p1 = wakeCentersYT[turbI, turb] + deltay - rotorDiameter[turbI]/2.
                p2 = wakeCentersYT[turbI, turb] + deltay + rotorDiameter[turbI]/2.

                wakeEffCoeffPerZone = GaussianAverage(max, mu, sigma, p1, p2)
                # wakeEffCoeffPerZone = GaussianMax(turbineY[turbI], max, mu, sigma)
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


def FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.5, 0.22, 0.5, 0.5, 0.620167, 5.500000])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""


    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    # me = np.array([-0.5, 0.22, 1.0])*np.array([1., 1., 0.65])
    # MU = np.array([0.5, 1.0, 5.5])*np.array([0., 0.63, 0.0])
    me = xdict['xvars'][0:3]
    MU = xdict['xvars'][3:6]
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction
    #
    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]


    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix,np.array([turbineX,turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0,nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines,nTurbines,3))
    for turb in range(0,nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(0,3):
                wakeDiametersT[turbI,turb,zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax,0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    wakeOverlapTRel = wakeOverlapT
    for turb in range(0,nTurbines):
        wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    keUncorrected = ke
    ke = np.zeros(nTurbines)
    for turb in range(0,nTurbines):
        s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
        ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0,deltax))),2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0,deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf,nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0, nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]
            deltay = wakeCentersYT[turbI, turb] - turbineY[turbI]

            if deltax > 0:
                mU = MU[1] / np.cos(aU*np.pi/180 + bU*yaw[turb])
                sigma = (wakeDiametersT[turbI, turb, 2] + rotorDiameter[turbI])/6.
                mu = wakeCentersYT[turbI, turb]
                max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)

                for zone in range(0, 4):
                    p1 = wakeCentersYT[turbI, turb] + deltay - 0.5*rotorDiameter[turbI]
                    p2 = wakeCentersYT[turbI, turb] + deltay + 0.5*rotorDiameter[turbI]
                    if zone == 3:
                        wakeEffCoeffPerZone += GaussianAverage(max, mu, sigma, p1, p2)*(1.-sum(wakeOverlapTRel[turbI, turb, :]))

                    elif wakeOverlapTRel[turbI, turb, zone] > 0.0:

                        # if wakeCentersYT[turbI, turb] + deltay - 0.5*rotorDiameter[turbI] < wakeCentersYT[turbI, turb] - wakeDiametersT[turbI, turb, zone]:
                        #     p1 = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone]
                        # if wakeCentersYT[turbI, turb] + deltay + 0.5*rotorDiameter[turbI] > wakeCentersYT[turbI, turb] + 0.5*rotorDiameter[turbI]:
                        #     p2 = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone]
                        # if zone > 0 and wakeOverlapTRel[turbI, turb, zone-1] > 0:
                        #     if wakeOverlapTRel[turbI, turb, zone-1] < 1 - 1E8:
                        #         if deltay > 0.0:
                        #             p1 = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone-1]
                        #         if deltay < 0.0:
                        #             p2 = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone-1]
                        #
                        # if zone > 0 and wakeOverlapTRel[turbI, turb, zone-1] >= 1 - 1E8:
                        #     p1p = wakeCentersYT[turbI, turb] - 0.5*wakeDiametersT[turbI, turb, zone]
                        #     p2p = wakeCentersYT[turbI, turb] + 0.5*wakeDiametersT[turbI, turb, zone]
                        #     Gave1 = GaussianAverage(max, mu, sigma, p1, p1p)
                        #     Gave2 = GaussianAverage(max, mu, sigma, p2, p2p)
                        #     Gave = 0.5*(Gave1 + Gave2)
                        #     wakeEffCoeffPerZone += Gave*wakeOverlapTRel[turbI, turb, zone]
                        # else:
                        wakeEffCoeffPerZone += GaussianAverage(max, mu, sigma, p1, p2)*wakeOverlapTRel[turbI, turb, zone]
                        # print wakeEffCoeffPerZone
                    # print "wake effective per zone Original", wakeEffCoeffPerZone
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


def FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.5, 0.22, 0.725083, 0.5, 1.0, 0.804150])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""

    # xdict = np.array(xdict)
    # print xdict
    pP = 1.88
    # ke = 0.065
    ke = 0.052
    # keCorrDA = 0
    # kd = 0.15
    # me = np.array([-0.5, 0.22, 1.0])*np.array([1., 1., 0.75])
    # MU = np.array([0.5, 1.0, 5.5])*np.array([0., 0., 0.1475])
    # me = xdict['xvars'][0:3][2]
    me = 1.0
    # MU = xdict['xvars'][3:6][2]
    # ad = -4.5
    # bd = -0.01
    # aU = 5.0
    # bU = 1.66

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
        # wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]), 2)*np.sin(yaw[turb])*4.*axialInd[turb]*(1.-axialInd[turb])
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]), 2)*np.sin(yaw[turb])*4.*axialInd[turb]*(1.-axialInd[turb])+2.0*np.pi/180.
        # wakeAngleInit = (0.6*axialInd[turb] + 1.)*yaw[turb]
        # for loc in range(0, nLocations):  # at velX-locations
        #     deltax = np.maximum(velX[loc]-turbineX[turb], 0)
        #     factor = (2*kd*deltax/rotorDiameter[turb])+1
        #     wakeCentersY[loc, turb] = turbineY[turb]
        #     wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + ad+bd*deltax  # rotation-induced deflection
        #     wakeCentersY[loc, turb] = wakeCentersY[loc, turb] + \
        #         (wakeAngleInit*(15*np.power(factor, 4)+np.power(wakeAngleInit, 2))/((30*kd*np.power(factor, 5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit, 4))/(30*kd))  # yaw-induced deflection
        for turbI in range(0, nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb], 0)
            kd = ke + 0.015
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI, turb] = turbineY[turb]
            # wakeCentersYT[turbI, turb] += ad+bd*deltax  # rotation-induced deflection
            # wakeCentersYT[turbI, turb] += ad  # rotation-induced deflection
            # wakeCentersYT[turbI, turb] = wakeCentersYT[turbI,turb] + \
            #     (wakeAngleInit*(15*np.power(factor, 4)+np.power(wakeAngleInit, 4))/((30*kd*np.power(factor, 5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit, 4))/(30*kd))  # yaw-induced deflection


            # ##############
            # calculate distance from wake cone apex to wake producing turbine
            x1 = 0.5*rotorDiameter[turb]/kd

            # calculate x position with cone apex as origin
            x = x1 + deltax

            # calculate wake offset due to yaw
            wakeCentersYT[turbI, turb] -= -wakeAngleInit*(x1**2)/x + x1*wakeAngleInit
            # ###############

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations, nTurbines))
    wakeDiametersT = np.zeros((nTurbines, nTurbines))
    for turb in range(0, nTurbines):
        for loc in range(0, nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            wakeDiameters[loc, turb] = rotorDiameter[turb]+2*ke*me*np.maximum(deltax, 0)
        for turbI in range(0, nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            # wakeDiametersT[turbI, turb] = np.maximum(rotorDiameter[turb]+2*ke*me*deltax, 0)
            wakeDiametersT[turbI, turb] = np.maximum(rotorDiameter[turb]+2*ke*deltax, 0)

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
    # for turb in range(0,nTurbines):
    #     mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
    #     for loc in range(0,nLocations):
    #         deltax = velX[loc] - turbineX[turb]
    #         radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
    #         axialIndAndNearRotor = 2*axialInd[turb]
    #
    #         if deltax > 0 and radiusLoc < wakeDiameters[loc, turb]/2.0:    # check if in zone 3
    #             reductionFactor = axialIndAndNearRotor * \
    #                 np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU)*np.maximum(0, deltax))), 2)
    #         elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
    #             reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0, deltax)/(rotorDiameter[turb]))/np.pi)
    #         else:
    #             reductionFactor = 0
    #         ws_array[loc] *= (1-reductionFactor)

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
                # mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb])
                # mU = MU
                sigma = (wakeDiametersT[turbI, turb] + rotorDiameter[turbI])/6.
                mu = wakeCentersYT[turbI, turb]
                # max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)
                max = np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*deltax), 2.0)
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


def FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter, yaw, Cp, axial_induction, turbineX, turbineY,
           xdict={'xvars': np.array([-0.493681, 0.196292, 0.587587, 0.442133, 1.110093, 5.537140, 3.0, 1.0])}, ws_positions=np.array([])):
    """Evaluates the FLORIS model and gives the FLORIS-predicted powers of the turbines at locations turbineX, turbineY,
    and, optionally, the FLORIS-predicted velocities at locations (velX,velY)"""
    # xdict={'xvars': np.array([-0.493681, 0.196292, 0.587587, 0.442133, 1.110093, 5.537140, 1E6, 1.0])} ## original FLORIS model newly tuned params
    # xdict={'xvars': np.array([-0.476202, 0.849977, 1.135163, 0.755217, 0.846327, 5.470013, 1.042578, 1.0])} best params, ignoring mag, for cos model
    # xdict={'xvars': np.array([-0.649728, 0.850570, 1.046345, 0.614197, 0.841373, 5.486359, 1.079179, 1.0])}
    # xdict={'xvars': np.array([-0.854513, 0.375026, 0.789352, 0.233813, 0.884157, 5.082038, 1.724035, 1.])}
    pP = 1.88
    ke = 0.065
    keCorrDA = 0
    kd = 0.15
    # me = np.array([-0.5, 0.22,1.0])
    # MU = np.array([0.5, 1.0, 5.5])
    me = xdict['xvars'][0:3]
    MU = xdict['xvars'][3:6]
    cos_spread = xdict['xvars'][6]
    cos_mag = xdict['xvars'][7]
    # cos_mag = 1.0
    ad = -4.5
    bd = -0.01
    aU = 5.0
    bU = 1.66

    # rename inputs and outputs
    Vinf = wind_speed                               # from standard FUSED-WIND GenericWindFarm component
    windDirection = wind_direction*np.pi/180        # from standard FUSED-WIND GenericWindFarm component
    rho = air_density

    rotorArea = np.pi*rotorDiameter**2/4.         # from standard FUSED-WIND GenericWindFarm component

    axialInd = axial_induction
    #
    # turbineX = positions[:, 0]
    # turbineY = positions[:, 1]


    yaw = yaw*np.pi/180.

    if ws_positions.any():
        velX = ws_positions[:, 0]
        velY = ws_positions[:, 1]
    else:
        velX = np.zeros([0,0])
        velY = np.zeros([0,0])

    # find size of arrays
    nTurbines = turbineX.size
    nLocations = velX.size

    # convert to downwind-crosswind coordinates
    rotationMatrix = np.array([(np.cos(-windDirection), -np.sin(-windDirection)),
                               (np.sin(-windDirection), np.cos(-windDirection))])
    turbineLocations = np.dot(rotationMatrix,np.array([turbineX,turbineY]))
    turbineX = turbineLocations[0]
    turbineY = turbineLocations[1]

    if velX.size>0:
        locations = np.dot(rotationMatrix,np.array([velX,velY]))
        velX = locations[0]
        velY = locations[1]

    # calculate y-location of wake centers
    wakeCentersY = np.zeros((nLocations,nTurbines))
    wakeCentersYT = np.zeros((nTurbines,nTurbines))
    for turb in range(0,nTurbines):
        wakeAngleInit = 0.5*np.power(np.cos(yaw[turb]),2)*np.sin(yaw[turb])*4*axialInd[turb]*(1-axialInd[turb])
        for loc in range(0,nLocations):  # at velX-locations
            deltax = np.maximum(velX[loc]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersY[loc,turb] = turbineY[turb]
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersY[loc,turb] = wakeCentersY[loc,turb] + \
                                     (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,2))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = np.maximum(turbineX[turbI]-turbineX[turb],0)
            factor = (2*kd*deltax/rotorDiameter[turb])+1
            wakeCentersYT[turbI,turb] = turbineY[turb]
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + ad+bd*deltax  # rotation-induced deflection
            wakeCentersYT[turbI,turb] = wakeCentersYT[turbI,turb] + \
                                        (wakeAngleInit*(15*np.power(factor,4)+np.power(wakeAngleInit,4))/((30*kd*np.power(factor,5))/rotorDiameter[turb]))-(wakeAngleInit*rotorDiameter[turb]*(15+np.power(wakeAngleInit,4))/(30*kd)) # yaw-induced deflection

    # calculate wake zone diameters at velX-locations
    wakeDiameters = np.zeros((nLocations,nTurbines,3))
    wakeDiametersT = np.zeros((nTurbines,nTurbines,3))
    for turb in range(0,nTurbines):
        for loc in range(0,nLocations):  # at velX-locations
            deltax = velX[loc]-turbineX[turb]
            for zone in range(0,3):
                wakeDiameters[loc,turb,zone] = rotorDiameter[turb]+2*ke*me[zone]*np.maximum(deltax,0)
        for turbI in range(0,nTurbines):  # at turbineX-locations
            deltax = turbineX[turbI]-turbineX[turb]
            for zone in range(0,3):
                wakeDiametersT[turbI,turb,zone] = np.maximum(rotorDiameter[turb]+2*ke*me[zone]*deltax,0)

    # calculate overlap areas at rotors
    # wakeOverlapT(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake
    # of turbine TURB with rotor of turbine TURBI
    wakeOverlapT = calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiametersT,wakeCentersYT)

    # make overlap relative to rotor area (maximum value should be 1)
    wakeOverlapTRel = wakeOverlapT
    for turb in range(0,nTurbines):
        wakeOverlapTRel[turb] = wakeOverlapTRel[turb]/rotorArea[turb]

    # array effects with full or partial wake overlap:
    # use overlap area of zone 1+2 of upstream turbines to correct ke
    # Note: array effects only taken into account in calculating
    # velocity deficits, in order not to over-complicate code
    # (avoid loops in calculating overlaps)

    keUncorrected = ke
    ke = np.zeros(nTurbines)
    for turb in range(0,nTurbines):
        s = np.sum(wakeOverlapTRel[turb,:,0]+wakeOverlapTRel[turb,:,1])
        ke[turb] = keUncorrected*(1+s*keCorrDA)

    # calculate velocities in full flow field (optional)
    ws_array = np.tile(Vinf,nLocations)
    for turb in range(0,nTurbines):
        mU = MU/np.cos(aU*np.pi/180+bU*yaw[turb])
        for loc in range(0,nLocations):
            deltax = velX[loc] - turbineX[turb]
            radiusLoc = abs(velY[loc]-wakeCentersY[loc,turb])
            axialIndAndNearRotor = 2*axialInd[turb]

            if deltax > 0 and radiusLoc < wakeDiameters[loc,turb,0]/2.0:    # check if in zone 1
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[0])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,1]/2.0:    # check if in zone 2
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[1])*np.maximum(0,deltax))),2)
            elif deltax > 0 and radiusLoc < wakeDiameters[loc,turb,2]/2.0:    # check if in zone 3
                reductionFactor = axialIndAndNearRotor*\
                                  np.power((rotorDiameter[turb]/(rotorDiameter[turb]+2*ke[turb]*(mU[2])*np.maximum(0,deltax))),2)
            elif deltax <= 0 and radiusLoc < rotorDiameter[turb]/2.0:     # check if axial induction zone in front of rotor
                reductionFactor = axialIndAndNearRotor*(0.5+np.arctan(2.0*np.minimum(0,deltax)/(rotorDiameter[turb]))/np.pi)
            else:
                reductionFactor = 0
            ws_array[loc] *= (1-reductionFactor)

    # find effective wind speeds at downstream turbines, then predict power downstream turbine
    velocitiesTurbines = np.tile(Vinf, nTurbines)

    for turbI in range(0,nTurbines):

        # find overlap-area weighted effect of each wake zone
        wakeEffCoeff = 0
        for turb in range(0,nTurbines):

            wakeEffCoeffPerZone = 0
            deltax = turbineX[turbI] - turbineX[turb]

            if deltax > 0:
                mU = MU / np.cos(aU*np.pi/180 + bU*yaw[turb])
                for zone in range(0,3):
                    rmax = cos_spread*0.5*(wakeDiametersT[turbI, turb, 2] + rotorDiameter[turbI])
                    cosFac = cos_mag*0.5*(1.0 + np.cos(np.pi*abs(wakeCentersYT[turbI, turb]-turbineY[turbI])/rmax))
                    wakeEffCoeffPerZone = wakeEffCoeffPerZone + np.power(cosFac*(rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU[zone]*deltax), 2.0) * wakeOverlapTRel[turbI,turb,zone]

                    # print "wake effective per zone Original", wakeEffCoeffPerZone
                wakeEffCoeff += np.power(axialInd[turb]*wakeEffCoeffPerZone, 2.0)

        # add gaussian curve to result
        mU = MU[2] / np.cos(aU*np.pi/180 + bU*yaw[turb])
        # sigma = (wakeDiametersT[turbI, turb, 2] + rotorDiameter[turbI])/6.
        # mu = wakeCentersYT[turbI, turb]
        # max = maxval #np.power((rotorDiameter[turb])/(rotorDiameter[turb]+2*ke[turb]*mU*deltax), 2.0)
        # print wakeEffCoeff, GaussianMax(turbineY[turbI], max, mu, sigma)
        # wakeEffCoeff += GaussianMax(turbineY[turbI], max, mu, sigma)

        wakeEffCoeff = (1 - 2 * np.sqrt(wakeEffCoeff))
        # print 'wakeEffCoeff original ', wakeEffCoeff
        # multiply the inflow speed with the wake coefficients to find effective wind speed at turbine
        velocitiesTurbines[turbI] *= wakeEffCoeff
        # velocitiesTurbines[turbI] += GaussianMax(turbineY[turbI], max, mu, sigma)
    # print 'velocitiesTurbines original ', velocitiesTurbines
    # find turbine powers
    wt_power = np.power(velocitiesTurbines, 3.0) * (0.5*rho*rotorArea*Cp*np.power(np.cos(yaw), pP))
    wt_power /= 1000  # in kW
    power = np.sum(wt_power)

    return velocitiesTurbines, wt_power, power, ws_array


def circleAreaForInt(R):
    f = 2*np.pi*R
    return f


def calcOverlapAreas(turbineX,turbineY,rotorDiameter,wakeDiameters,wakeCenters):
    """calculate overlap of rotors and wake zones (wake zone location defined by wake center and wake diameter)
    turbineX,turbineY is x,y-location of center of rotor

    wakeOverlap(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake of turbine TURB with rotor of downstream turbine
    TURBI"""

    nTurbines = turbineY.size

    wakeOverlap = np.zeros((nTurbines,nTurbines,3))

    for turb in range(0,nTurbines):
        for turbI in range(0,nTurbines):
            if turbineX[turbI] > turbineX[turb]:
                OVdYd = wakeCenters[turbI,turb]-turbineY[turbI]
                OVr = rotorDiameter[turbI]/2
                for zone in range(0,3):
                    OVR = wakeDiameters[turbI,turb,zone]/2
                    OVdYd = abs(OVdYd)
                    if OVdYd != 0:
                        OVL = (-np.power(OVr,2.0)+np.power(OVR,2.0)+np.power(OVdYd,2.0))/(2.0*OVdYd)
                    else:
                        OVL = 0

                    OVz = np.power(OVR,2.0)-np.power(OVL,2.0)

                    if OVz > 0:
                        OVz = np.sqrt(OVz)
                    else:
                        OVz = 0

                    if OVdYd < (OVr+OVR):
                        if OVL < OVR and (OVdYd-OVL) < OVr:
                            wakeOverlap[turbI,turb,zone] = np.power(OVR,2.0)*np.arccos(OVL/OVR) + np.power(OVr,2.0)*np.arccos((OVdYd-OVL)/OVr) - OVdYd*OVz
                        elif OVR > OVr:
                            wakeOverlap[turbI,turb,zone] = np.pi*np.power(OVr,2.0)
                        else:
                            wakeOverlap[turbI,turb,zone] = np.pi*np.power(OVR,2.0)
                    else:
                        wakeOverlap[turbI,turb,zone] = 0

    for turb in range(0,nTurbines):
        for turbI in range(0,nTurbines):
            wakeOverlap[turbI,turb,2] = wakeOverlap[turbI,turb,2]-wakeOverlap[turbI,turb,1]
            wakeOverlap[turbI,turb,1] = wakeOverlap[turbI,turb,1]-wakeOverlap[turbI,turb,0]

    return wakeOverlap


def gaussian(x, mu, sigma):

    f = (1./(sigma*np.sqrt(2.*np.pi)))*np.exp((-(x-mu)**2)/(2.*sigma**2))

    return f


def GaussianExactSector(R, r, d, C, Dw):
    """
    :param R: radial distance from wake center
    :param r: radius of wind turbine rotor
    :param d: horizontal distance between hub and wake center
    :param C: Ci(x) defining wake deficit
    :param Dw: width of the wake (should be 4*std.-dev.)
    :return: product of area and wake deficit for use with integration
    """
    f = 2.0*R*np.arccos((d**2+R**2-r**2)/(2*d*R))

    sigma = Dw/4.
    mu = 0.
    g = C*np.exp((R-mu)**2/(2.*sigma**2))

    return f*g


def LinearizedExactSector(R, r, d, C, Dw):
    """
    :param R: radial distance from wake center
    :param r: radius of wind turbine rotor
    :param d: horizontal distance between hub and wake center
    :param C: numpy array of Ci(x, R) defining wake deficit
    :param Dw: width of the wake (should be 4*std.-dev.)
    :return: product of area and wake deficit for use with integration
    """
    # print 'vals: ', d, R, r

    OVdYd = d
    OVr = r
    OVR = R
    OVdYd = abs(OVdYd)
    if OVdYd != 0:
        OVL = (-np.power(OVr, 2.0)+np.power(OVR, 2.0)+np.power(OVdYd, 2.0))/(2.0*OVdYd)
    else:
        OVL = 0

    if OVdYd < (OVr+OVR):
        if OVL < OVR and (OVdYd-OVL) < OVr:
            f = 2.*OVR*np.arccos(OVL/OVR)
        elif OVR > OVr:
            f = 2.*np.pi*OVr
        else:
            f = 2.*np.pi*OVR
    else:
        f = 0.0

    # if d + R > r:
    #     f = 2.0*R*np.arccos((d**2+R**2-r**2)/(2.*d*R))
    # else:
    #     f = 2.0*R*np.pi

    if R <= Dw[1]/2.:
        g = 2.*((C[1]-C[0])/(Dw[1]-Dw[0]))*(R-Dw[0]/2.)+C[0]
        # print 'g, R, C ', g, R, C
        # print 10000000000000000000000
    elif R <= Dw[2]/2.:
        g = 2.*((C[2]-C[1])/(Dw[2]-Dw[1]))*(R-Dw[1]/2.)+C[1]
        # print 20000000000000000000000
    elif R <= Dw[3]/2.:
        g = 2.*((C[3]-C[2])/(Dw[3]-Dw[2]))*(R-Dw[2]/2.)+C[2]
        # print 30000000000000000000000
    elif R <= Dw[4]/2.:
        g = 2.*((C[4]-C[3])/(Dw[4]-Dw[3]))*(R-Dw[3]/2.)+C[3]
        # print 40000000000000000000000
    else:
        g = 0.0
        # print 50000000000000000000000
    # g = 1.
    # print 'g, R, C ', g, R, C
    # print 'g ', g

    return f*g


def LinearizedHeight(R, r, d, C, Dw):
    """
    :param R: radial distance from wake center
    :param r: radius of wind turbine rotor
    :param d: horizontal distance between hub and wake center
    :param C: numpy array of Ci(x, R) defining wake deficit
    :param Dw: width of the wake (should be 4*std.-dev.)
    :return: product of area and wake deficit for use with integration
    """
    d = abs(d)
    hR = np.sqrt(r**2-(d-R)**2)
    hr = np.sqrt()
    # print 'vals: ', d, R, r

    OVdYd = d
    OVr = r
    OVR = R
    OVdYd = abs(OVdYd)
    if OVdYd != 0:
        OVL = (-np.power(OVr, 2.0)+np.power(OVR, 2.0)+np.power(OVdYd, 2.0))/(2.0*OVdYd)
    else:
        OVL = 0

    if OVdYd < (OVr+OVR):
        if OVL < OVR and (OVdYd-OVL) < OVr:
            f = 2.*OVR*np.arccos(OVL/OVR)
        # elif OVR > OVr:
        #     f = 2.*np.pi*OVr
        else:
            f = 2.*np.pi*OVR
    else:
        f = 0.0

    # if d + R > r:
    #     f = 2.0*R*np.arccos((d**2+R**2-r**2)/(2.*d*R))
    # else:
    #     f = 2.0*R*np.pi

    if R <= Dw[1]/2.:
        g = 2.*((C[1]-C[0])/(Dw[1]-Dw[0]))*(R-Dw[0]/2.)+C[0]
        # print 'g, R, C ', g, R, C
        # print 10000000000000000000000
    elif R <= Dw[2]/2.:
        g = 2.*((C[2]-C[1])/(Dw[2]-Dw[1]))*(R-Dw[1]/2.)+C[1]
        # print 20000000000000000000000
    elif R <= Dw[3]/2.:
        g = 2.*((C[3]-C[2])/(Dw[3]-Dw[2]))*(R-Dw[2]/2.)+C[2]
        # print 30000000000000000000000
    elif R <= Dw[4]/2.:
        g = 2.*((C[4]-C[3])/(Dw[4]-Dw[3]))*(R-Dw[3]/2.)+C[3]
        # print 40000000000000000000000
    else:
        g = 0.0
        # print 50000000000000000000000
    # g = 1.
    # print 'g, R, C ', g, R, C
    # print 'g ', g

    return f*g


def LinearizedSingleSector(r, d, C, Dw, R1, R2):
    """
    :param R: radial distance from wake center
    :param r: radius of wind turbine rotor
    :param d: horizontal distance between hub and wake center
    :param C: numpy array of Ci(x, R) defining wake deficit
    :param Dw: width of the wake (should be 4*std.-dev.)
    :return: integral of area and wake deficit using segment angle at average R value
    """

    R = (R2-R1)/2.

    theta = np.arccos((d**2+R**2-r**2)/(2.*d*R))

    if R <= Dw[1]:
        CC = (C[1]-C[0])/(Dw[1]-Dw[0])
        DD = Dw[0]/2.
        Q = C[0]
    elif R <= Dw[2]:
        CC = (C[2]-C[1])/(Dw[2]-Dw[1])
        DD = Dw[1]/2.
        Q = C[1]
    elif R <= Dw[3]:
        CC = (C[3]-C[2])/(Dw[3]-Dw[2])
        DD = Dw[2]/2.
        Q = C[2]
    elif R <= Dw[4]:
        CC = (C[4]-C[3])/(Dw[4]-Dw[3])
        DD = Dw[3]/2.
        Q = C[3]

    f = (4./3.)*theta*CC*(R2**3-R1**3)+theta*(-2*CC*DD+Q)*(R2**2-R1**2)

    return f


def GaussianAverage(max, mu, sigma, p1, p2):
    """
    Computes the average value of the standard normal distribution on the interval [p1, p2]
    :param max: max value (peak) of the gaussian normal distribution
    :param mu: mean of the standard normal distribution
    :param sigma: standard distribution of the standard normal distribution
    :param p1: first point of the interval of interest
    :param p2: second point of the interval of interest
    :return: average value of the standard normal distribution on given interval
    """

    val1 = 0.5*max*np.sqrt(2.*np.pi)*sigma*sp.special.erf(0.5*np.sqrt(2.)*(p1-mu)/sigma)
    val2 = 0.5*max*np.sqrt(2.*np.pi)*sigma*sp.special.erf(0.5*np.sqrt(2.)*(p2-mu)/sigma)
    average = (val2-val1)/(p2-p1)

    return average


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


if __name__ == "__main__":

    nTurbines = 2
    nDirections = 1

    rotorDiameter = 126.4
    rotorArea = np.pi*rotorDiameter*rotorDiameter/4.0
    axialInduction = 1.0/3.0
    CP = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CP =0.768 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    CT = 4.0*axialInduction*(1.0-axialInduction)
    generator_efficiency = 0.944

    # Define turbine characteristics
    axialInduction = np.array([axialInduction, axialInduction])
    rotorDiameter = np.array([rotorDiameter, rotorDiameter])
    # rotorDiameter = np.array([rotorDiameter, 0.0001*rotorDiameter])
    yaw = np.array([0., 0.])

    # Define site measurements
    wind_direction = 30.
    wind_speed = 8.    # m/s
    air_density = 1.1716

    Ct = np.array([CT, CT])
    Cp = np.array([CP, CP])

    ICOWESdata = loadmat('../../data/YawPosResults.mat')
    yawrange = ICOWESdata['yaw'][0]

    FLORISpower = list()
    # FLORISpowerStat = list()
    # FLORISpowerLES = list()
    FLORISpowerGAZ = list()
    FLORISpowerGAR = list()
    FLORISpowerGARZ = list()
    FLORISpowerGH = list()
    FLORISpowerCos = list()
    FLORISgradient = list()
    FLORISvelocity = list()
    # FLORISvelocityStat = list()
    # FLORISvelocityLES = list()
    FLORISvelocityGAZ = list()
    FLORISvelocityGAR = list()
    FLORISvelocityGARZ = list()
    FLORISvelocityGH = list()
    FLORISvelocityCos = list()

    for yaw1 in yawrange:

        # Defube turbine locations and orientation
        turbineX = np.array([1118.1, 1881.9])
        turbineY = np.array([1279.5, 1720.5])

        # print np.sqrt((turbineY[0]-turbineY[1])**2+(turbineX[0]-turbineX[1])**2)/rotorDiameter[0] # print downwind distance

        yaw = np.array([yaw1, 0.0])

        # positions = np.transpose(np.array([turbineX, turbineY]))

        # print positions

        # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)
        t1 = time.time()
        for i in range(0, 1):
            # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
            #                                                                                 air_density, rotorDiameter,
            #                                                                                 yaw, Cp, axialInduction,
            #                                                                                 turbineX, turbineY)
            velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                                   yaw, Cp, axialInduction, turbineX, turbineY)

        t2 = time.time()

        # print 'time LES: ', t2-t1
        print 'time GAZ: ', t2-t1

        t1 = time.time()
        for i in range(0, 1):

            velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                                   yaw, Cp, axialInduction, turbineX, turbineY)

        t2 = time.time()

        # print 'time LES: ', t2-t1
        print 'time GARZ: ', t2-t1

        t1 = time.time()
        for i in range(0, 1):

            velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                                   yaw, Cp, axialInduction, turbineX, turbineY)

        t2 = time.time()
        print 'time GAR: ', t2-t1

        t1 = time.time()
        for i in range(0, 1):

            velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                                   yaw, Cp, axialInduction, turbineX, turbineY)

        t2 = time.time()
        print 'time GH: ', t2-t1

        t1 = time.time()
        for i in range(0, 1):

            velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                                   yaw, Cp, axialInduction, turbineX, turbineY)

        t2 = time.time()
        print 'time Cos: ', t2-t1

        t1 = time.time()
        for i in range(0, 1):
            velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                   yaw, Cp, axialInduction, turbineX, turbineY,)
        t2 = time.time()

        print 'time original: ', t2-t1

        FLORISpower.append(list(wt_power))
        # FLORISpowerStat.append(list(wt_power_stat))
        # FLORISpowerLES.append(list(wt_power_LES))
        FLORISpowerGAZ.append(list(wt_power_GAZ))
        FLORISpowerGAR.append(list(wt_power_GAR))
        FLORISpowerGARZ.append(list(wt_power_GARZ))
        FLORISpowerGH.append(list(wt_power_GH))
        FLORISpowerCos.append(list(wt_power_Cos))

    FLORISpower = np.array(FLORISpower)
    # FLORISpowerStat = np.array(FLORISpowerStat)
    # FLORISpowerLES = np.array(FLORISpowerLES)
    FLORISpowerGAZ = np.array(FLORISpowerGAZ)
    FLORISpowerGAR = np.array(FLORISpowerGAR)
    FLORISpowerGARZ = np.array(FLORISpowerGARZ)
    FLORISpowerGH = np.array(FLORISpowerGH)
    FLORISpowerCos = np.array(FLORISpowerCos)

    SOWFApower = np.array([ICOWESdata['yawPowerT1'][0],ICOWESdata['yawPowerT2'][0]]).transpose()/1000.

    fig, axes = plt.subplots(ncols=2, nrows=2, sharey=False)
    axes[0, 0].plot(yawrange.transpose(), FLORISpower[:, 0], 'b', yawrange.transpose(), SOWFApower[:, 0], 'bo')
    axes[0, 0].plot(yawrange.transpose(), FLORISpower[:, 1], 'b', yawrange.transpose(), SOWFApower[:, 1], 'bo')
    axes[0, 0].plot(yawrange.transpose(), FLORISpower[:, 0]+FLORISpower[:, 1], 'k-', yawrange.transpose(), SOWFApower[:, 0]
                 + SOWFApower[:, 1], 'ko')
    # axes[0].plot(yawrange, FLORISpowerStat[:, 1], 'g-')
    # axes[0].plot(yawrange, FLORISpowerLES[:, 1], 'c-')
    # axes[0, 0].plot(yawrange, FLORISpowerGAZ[:, 1], 'm-')
    # axes[0, 0].plot(yawrange, FLORISpowerGAR[:, 1], 'c-')
    # axes[0, 0].plot(yawrange, FLORISpowerGARZ[:, 1], 'g-')
    axes[0, 0].plot(yawrange, FLORISpowerGH[:, 1], 'y-')
    axes[0, 0].plot(yawrange, FLORISpowerCos[:, 1], 'r')
    axes[0, 0].set_xlabel('yaw angle (deg.)')
    axes[0, 0].set_ylabel('Power (kW)')
    error_turbine2 = np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:, 1]))
    posrange = ICOWESdata['pos'][0]

    yaw = np.array([0.0, 0.0])
    FLORISpower = list()
    # FLORISpowerStat = list()
    # FLORISpowerLES = list()
    FLORISpowerGAZ = list()
    FLORISpowerGAR = list()
    FLORISpowerGARZ = list()
    FLORISpowerGH = list()
    FLORISpowerCos = list()


    for pos2 in posrange:
        # Define turbine locations and orientation
        effUdXY = 0.523599

        Xinit = np.array([1118.1, 1881.9])
        Yinit = np.array([1279.5, 1720.5])
        XY = np.array([Xinit, Yinit]) + np.dot(np.array([[np.cos(effUdXY), -np.sin(effUdXY)],
                                                                [np.sin(effUdXY), np.cos(effUdXY)]]),
                                                      np.array([[0., 0], [0, pos2]]))

        turbineX = XY[0, :]
        turbineY = XY[1, :]
        # print 'y pos:', turbineX, turbineY
        yaw = np.array([0.0, 0.0])

        # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)

        # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)

        velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
                                                                               yaw, Cp, axialInduction, turbineX, turbineY,)


        velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        # print 'power = ', myFloris.root.dir0.unknowns['wt_power']
        FLORISpower.append(list(wt_power))
        # FLORISpowerStat.append(list(wt_power_stat))
        FLORISpowerGAZ.append(list(wt_power_GAZ))
        FLORISpowerGAR.append(list(wt_power_GAR))
        FLORISpowerGARZ.append(list(wt_power_GARZ))
        FLORISpowerGH.append(list(wt_power_GH))
        FLORISpowerCos.append(list(wt_power_Cos))

    FLORISpower = np.array(FLORISpower)
    # FLORISpowerStat = np.array(FLORISpowerStat)
    # FLORISpowerLES = np.array(FLORISpowerLES)
    FLORISpowerGAZ = np.array(FLORISpowerGAZ)
    FLORISpowerGAR = np.array(FLORISpowerGAR)
    FLORISpowerGARZ = np.array(FLORISpowerGARZ)
    FLORISpowerGH = np.array(FLORISpowerGH)
    FLORISpowerCos = np.array(FLORISpowerCos)


    SOWFApower = np.array([ICOWESdata['posPowerT1'][0], ICOWESdata['posPowerT2'][0]]).transpose()/1000.

    error_turbine2 += np.sum(np.abs(FLORISpower[:, 1] - SOWFApower[:,1]))

    # print error_turbine2

    axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpower[:, 0], 'b', posrange/rotorDiameter[0], SOWFApower[:, 0], 'bo')
    axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpower[:, 1], 'b', posrange/rotorDiameter[0], SOWFApower[:, 1], 'bo')
    axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpower[:, 0]+FLORISpower[:, 1], 'k-', posrange/rotorDiameter[0], SOWFApower[:, 0]+SOWFApower[:, 1], 'ko')
    # axes[1].plot(posrange, FLORISpowerStat[:, 1], 'g-')
    # axes[1].plot(posrange/rotorDiameter[0], FLORISpowerLES[:, 1], 'c-')
    # axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpowerGAZ[:, 1], 'm-')
    # axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpowerGAR[:, 1], 'c-')
    # axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpowerGARZ[:, 1], 'g-')
    axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpowerGH[:, 1], 'y-')
    axes[0, 1].plot(posrange/rotorDiameter[0], FLORISpowerCos[:, 1], 'r')

    axes[0, 1].set_xlabel('y/D')
    axes[0, 1].set_ylabel('Power (kW)')

    posrange = np.linspace(-3.*rotorDiameter[0], 3.*rotorDiameter[0], num=1000)
    yaw = np.array([0.0, 0.0])
    wind_direction = 0.0
    FLORISvelocity = list()
    # FLORISvelocityStat = list()
    FLORISvelocityLES = list()
    for pos2 in posrange:

        turbineX = np.array([0, 7.*rotorDiameter[0]])
        turbineY = np.array([0, pos2])
        # print turbineX, turbineY

        # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)

        # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)

        velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)




        velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        # print 'power = ', myFloris.root.dir0.unknowns['wt_power']

        FLORISvelocity.append(list(velocitiesTurbines))
        # FLORISvelocityStat.append(list(velocitiesTurbines_stat))
        # FLORISvelocityLES.append(list(velocitiesTurbines_LES))
        FLORISvelocityGAZ.append(list(velocitiesTurbines_GAZ))
        FLORISvelocityGAR.append(list(velocitiesTurbines_GAR))
        FLORISvelocityGARZ.append(list(velocitiesTurbines_GARZ))
        FLORISvelocityGH.append(list(velocitiesTurbines_GH))
        FLORISvelocityCos.append(list(velocitiesTurbines_Cos))


    FLORISvelocity = np.array(FLORISvelocity)
    # FLORISvelocityStat = np.array(FLORISvelocityStat)
    # FLORISvelocityLES = np.array(FLORISvelocityLES)
    FLORISvelocityGAZ = np.array(FLORISvelocityGAZ)
    FLORISvelocityGAR = np.array(FLORISvelocityGAR)
    FLORISvelocityGARZ = np.array(FLORISvelocityGARZ)
    FLORISvelocityGH = np.array(FLORISvelocityGH)
    FLORISvelocityCos = np.array(FLORISvelocityCos)


    axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocity[:, 1], 'b', label='Floris Original')
    # axes[2].plot(posrange, FLORISvelocityStat[:, 1], 'g', label='added gaussian')
    # axes[2].plot(posrange/rotorDiameter[0], FLORISvelocityLES[:, 1], 'c', label='Linear Exact Sector')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGAZ[:, 1], 'm', label='GAZ')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGAR[:, 1], 'c', label='GAR')
    # axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGARZ[:, 1], 'g', label='GARZ')
    axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityGH[:, 1], 'y', label='GH')
    axes[1, 0].plot(posrange/rotorDiameter[0], FLORISvelocityCos[:, 1], 'r', label='Cos')

    axes[1, 0].set_xlabel('y/D')
    axes[1, 0].set_ylabel('Velocity (m/s)')
    # plt.legend()
    # plt.show()

    posrange = np.linspace(-1.*rotorDiameter[0], 30.*rotorDiameter[0], num=2000)
    yaw = np.array([0.0, 0.0])
    wind_direction = 0.0
    FLORISvelocity = list()
    # FLORISvelocityStat = list()
    FLORISvelocity = list()
    # FLORISvelocityStat = list()
    # FLORISvelocityLES = list()
    FLORISvelocityGAZ = list()
    FLORISvelocityGAR = list()
    FLORISvelocityGARZ = list()
    FLORISvelocityGH = list()
    FLORISvelocityCos = list()
    for pos2 in posrange:

        turbineX = np.array([0, pos2])
        turbineY = np.array([0, 0])
        # print turbineX, turbineY

        # velocitiesTurbines_stat, wt_power_stat, power_stat, ws_array_stat = FLORIS_Stat(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)

        # velocitiesTurbines_LES, wt_power_LES, power_LES = FLORIS_LinearizedExactSector(wind_speed, wind_direction,
        #                                                                                 air_density, rotorDiameter,
        #                                                                                 yaw, Cp, axialInduction,
        #                                                                                 turbineX, turbineY)

        velocitiesTurbines_GAZ, wt_power_GAZ, power_GAZ, ws_array_GAZ = FLORIS_GaussianAveZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GAR, wt_power_GAR, power_GAR, ws_array_GAR = FLORIS_GaussianAveRotor(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GARZ, wt_power_GARZ, power_GARZ, ws_array_GARZ = FLORIS_GaussianAveRotorZones(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_GH, wt_power_GH, power_GH, ws_array_GH = FLORIS_GaussianHub(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        velocitiesTurbines_Cos, wt_power_Cos, power_Cos, ws_array_Cos = FLORIS_Cos(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)





        velocitiesTurbines, wt_power, power, ws_array = FLORIS(wind_speed, wind_direction, air_density, rotorDiameter,
                                                               yaw, Cp, axialInduction, turbineX, turbineY,)

        # print 'power = ', myFloris.root.dir0.unknowns['wt_power']

        FLORISvelocity.append(list(velocitiesTurbines))
        # FLORISvelocityStat.append(list(velocitiesTurbines_stat))
        # FLORISvelocityLES.append(list(velocitiesTurbines_LES))
        FLORISvelocityGAZ.append(list(velocitiesTurbines_GAZ))
        FLORISvelocityGAR.append(list(velocitiesTurbines_GAR))
        FLORISvelocityGARZ.append(list(velocitiesTurbines_GARZ))
        FLORISvelocityGH.append(list(velocitiesTurbines_GH))
        FLORISvelocityCos.append(list(velocitiesTurbines_Cos))


    FLORISvelocity = np.array(FLORISvelocity)
    # FLORISvelocityStat = np.array(FLORISvelocityStat)
    # FLORISvelocityLES = np.array(FLORISvelocityLES)
    FLORISvelocityGAZ = np.array(FLORISvelocityGAZ)
    FLORISvelocityGAR = np.array(FLORISvelocityGAR)
    FLORISvelocityGARZ = np.array(FLORISvelocityGARZ)
    FLORISvelocityGH = np.array(FLORISvelocityGH)
    FLORISvelocityCos = np.array(FLORISvelocityCos)


    axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocity[:, 1], 'b', label='Floris Original')
    # axes[2].plot(posrange, FLORISvelocityStat[:, 1], 'g', label='added gaussian')
    # axes[2].plot(posrange/rotorDiameter[0], FLORISvelocityLES[:, 1], 'c', label='Linear Exact Sector')
    # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGAZ[:, 1], 'm', label='GAZ')
    # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGAR[:, 1], 'c', label='GAR')
    # axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGARZ[:, 1], 'g', label='GARZ')
    axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityGH[:, 1], 'y', label='GH')
    axes[1, 1].plot(posrange/rotorDiameter[0], FLORISvelocityCos[:, 1], 'r', label='Cos')
    axes[1, 1].plot(np.array([7, 7]), np.array([2, 8]), '--k', label='tuning point')
    print min(FLORISvelocity[:, 1])
    plt.xlabel('x/D')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc=4)
    plt.show()
