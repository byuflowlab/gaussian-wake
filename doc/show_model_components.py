import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from openmdao.api import Problem

from wakeexchange.OptimizationGroups import OptAEP
from wakeexchange.gauss import gauss_wrapper, add_gauss_params_IndepVarComps

def get_x0(rotor_diameter, yaw, Ct, alpha, beta, I):

    x0 = rotor_diameter * (np.cos(yaw) * (1.0 + np.sqrt(1.0 - Ct)) /
                          (np.sqrt(2.0) * (alpha * I + beta * (1.0 - np.sqrt(1.0 - Ct)))))
    print 'x0 = ', x0/rotor_diameter
    return x0

def get_sigmayz(x, x0, yaw, rotor_diameter, ky, kz):

    deltax0 = x - x0

    print 'deltax0 = ', deltax0

    sigmay = rotor_diameter * (ky * deltax0 / rotor_diameter
                              + np.cos(yaw) / np.sqrt(8.0))
    sigmaz = rotor_diameter * (kz * deltax0 / rotor_diameter + 1.0 / np.sqrt(8.0))

    return sigmay, sigmaz

def get_wake_deflection(yaw, rotor_diameter, Ct, ky, kz, sigmay, sigmaz):

    theta_c_0 = 0.3 * yaw * (1.0 - np.sqrt(1.0 - Ct * np.cos(yaw))) / np.cos(yaw)

    wake_offset = rotor_diameter * (theta_c_0 * x0 / rotor_diameter + (theta_c_0 / 14.7)
                                    *np.sqrt(np.cos(yaw) / (ky * kz * Ct)) *
                                    (2.9 + 1.3 * np.sqrt(1.0 - Ct) - Ct) *
                                    np.log(((1.6 + np.sqrt(Ct)) *
                                            (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                                                           (np.cos(yaw) *
                                                            rotor_diameter ** 2)) -
                                             np.sqrt(Ct))) /
                                           ((1.6 - np.sqrt(Ct)) *
                                            (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
                                                           (np.cos(yaw) *
                                                            rotor_diameter ** 2)) +
                                             np.sqrt(Ct)))))

    return wake_offset

def get_velocity_def(Yi, Yt, Zi, Zt, yaw, wake_offset, Ct, sigmay, sigmaz, rotor_diameter):

    deltay = Yi - (Yt + wake_offset)

    if Ct * np.cos(yaw)/(8.0 * sigmay * sigmaz /
             (rotor_diameter ** 2)) > 1:
        deltav = 1.

    else:

        deltav = ((1.0 - np.sqrt(1.0 - Ct * np.cos(yaw) /
                                              (8.0 * sigmay * sigmaz /
                                              (rotor_diameter ** 2)))) *
                               np.exp(-0.5 * ((deltay) / sigmay) ** 2) *
                               np.exp(-0.5 * ((Zi - Zt) / sigmaz) ** 2))

    print np.sqrt(1.0 - Ct * np.cos(yaw) /
            (8.0 * sigmay * sigmaz /
             (rotor_diameter ** 2))) , Ct * np.cos(yaw)/(8.0 * sigmay * sigmaz /
             (rotor_diameter ** 2))

    return deltav

if __name__ == "__main__":

    rotorDiameter = 126.4
    axialInduction = 1.0 / 3.0  # used only for initialization
    hub_height = 90.0
    Ct = 4.0 * axialInduction * (1.0 - axialInduction)
    Cp = 0.7737 / 0.944 * 4.0 * 1.0 / 3.0 * np.power((1 - 1.0 / 3.0), 2)
    yaw = 20.0*np.pi/180.
    Ct = Ct*np.cos(yaw)**2

    ky = 0.022
    kz = 0.022
    alpha = 2.32
    beta = 0.154
    I = 0.075  # .075

    XI = np.arange(0., 10.*rotorDiameter, 1)
    index = 0
    sigmayA = np.zeros_like(XI)
    sigmazA = np.zeros_like(XI)
    wake_offsetA = np.zeros_like(XI)
    deltavA = np.zeros_like(XI)

    x0 = get_x0(rotorDiameter, yaw, Ct, alpha, beta, I)
    sigmay0, sigmaz0 = get_sigmayz(x0, x0, yaw, rotorDiameter, ky, kz)
    wake_offset0 = get_wake_deflection(yaw, rotorDiameter, Ct, ky, kz, sigmay0, sigmaz0)
    deltav0 = get_velocity_def(wake_offset0, 0., hub_height, hub_height, yaw, wake_offset0, Ct, sigmay0, sigmaz0, rotorDiameter)
    ur = Ct*np.cos(yaw)/(2.*(1.-np.sqrt(1.-Ct*np.cos(yaw))))
    u0 = 1. - np.sqrt(1-Ct)
    u = 0.5*(1.+u0)
    u_ai = (1.-axialInduction)*1.

    for x in XI:
        sigmayA[index], sigmazA[index] = get_sigmayz(x, x0, yaw, rotorDiameter, ky, kz)
        wake_offsetA[index] = get_wake_deflection(yaw, rotorDiameter, Ct, ky, kz, sigmayA[index], sigmazA[index])
        deltavA[index] = get_velocity_def(wake_offsetA[index], 0., hub_height, hub_height, yaw, wake_offsetA[index], Ct, sigmayA[index], sigmazA[index], rotorDiameter)
        index += 1

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)

    XI /= rotorDiameter
    x0 /= rotorDiameter
    sigmayA /= rotorDiameter
    sigmazA /= rotorDiameter

    ax1.plot(XI, sigmayA, linestyle='--', label="$\sigma_{y}$")
    ax1.plot(XI, sigmazA, linestyle=':', label="$\sigma_{z}$")
    ax1.set_ylabel('Wake Spread ($\sigma / D_r$)')
    ax1.legend(loc=4)

    ax2.plot(XI, wake_offsetA, label='$\delta$')
    ax2.legend(loc=4)
    ax2.set_ylabel('Distance (m)')

    ax3.plot(XI, deltavA, label='$\Delta V$')
    ax3.plot(XI, np.ones_like(XI)*ur, label='$u_r$')
    ax3.plot(XI, np.ones_like(XI)*u0, label='$u_0$', linestyle=':')
    ax3.plot(XI, np.ones_like(XI)*u, label='$u_{ave}$', linestyle='--')
    ax3.plot(XI, np.ones_like(XI)*u_ai, label='$u_{ai}$', linestyle='-.')
    ax3.scatter(x0, deltav0, label='$\Delta V_{o}$')
    ax3.legend(loc=4)
    ax3.set_xlabel('Rotor Diameters')
    ax3.set_ylabel('Velocity Deficit ($1 - V/V_{0}$)')

    fig.suptitle('For Yaw = %.01f$^o$' % (yaw*180./np.pi))

    plt.show()


