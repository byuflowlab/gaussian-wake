import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


def approx(x, wake_spread, yaw, Ct, Rd):

    # calculate initial wake angle
    initial_wake_angle = 3.0*np.pi/180. + 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2

    # calculate distance from wake cone apex to wake producing turbine
    x1 = Rd/np.tan(wake_spread)

    x += x1

    # calculate wake offset due to yaw
    # deltaY = initial_wake_angle*((x**3)/(x1**2) - 4.*(x**2)/x1 + 6.*x - 3.*x1)
    deltaY = -initial_wake_angle*(x1**2)/x + x1*initial_wake_angle

    return deltaY + 4.5


def exact(x, wake_spread, yaw, Ct, Rd, a=0, burton_correction=False):

    def exact_func(x, wake_spread, yaw, Ct, Rd, a=0, burton_correction=False):
        if burton_correction:
            kappa0 = yaw
            initial_wake_angle = (0.6*a + 1.)*kappa0
        else:
            initial_wake_angle = 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2
        x1 = Rd/np.tan(wake_spread)
        x += x1
        wake_angle = initial_wake_angle*(x1/x)**2
        return np.tan(wake_angle)

    deltaY, _ = quad(exact_func, 0.0, x, args=(wake_spread, yaw, Ct, Rd, a, burton_correction))

    return deltaY


def floris(x, wake_spread, yaw, Ct, Rd):

    D = 2.*Rd
    initial_wake_angle = 0.5*Ct*np.sin(yaw)*np.cos(yaw)**2
    x1 = Rd/np.tan(wake_spread)
    kd = (D-1.)/x1

    deltaY = 15.*(2.*kd*x/D + 1.)**4
    deltaY += initial_wake_angle**2
    deltaY /= (30.*kd/D)*(2.*kd*x/D + 1.)**5
    deltaY -= D*(15. + initial_wake_angle**2)/(30.*kd)
    deltaY *= initial_wake_angle

    return deltaY


def ypower(x, m, n, nu, w, yaw, Ct, Rd):
    """

    :param x:
    :param m: assumes m < 0.5
    :param n:
    :param nu:
    :param w:
    :param yaw:
    :param Ct:
    :param Rd:
    :return:
    """
    if m >= 0.5:
        raise ValueError('m must be less than 0.5')

    D = 2.*Rd
    w *= D
    xi = 0.5*Ct*np.power(np.cos(yaw), 2)*np.sin(yaw)
    deltay = -xi*(np.power(x, -2.*m+1.)*np.power(D, 2.+2.*m))/((2.*m-1.)*w**2)
    return deltay

def bastankhah(x, k, yaw, Ct, Rd):

    m = 0.
    n = 2.

    rotor_diameter = 2*Rd*np.cos(yaw)
    beta = 0.5*((1.+np.sqrt(1.-Ct))/np.sqrt(1.-Ct))
    epsilon = 0.2*np.sqrt(beta)
    xi = 0.5*Ct*np.power(np.cos(yaw), 2)*np.sin(yaw)
    # deltay = (0.25)*xi*rotor_diameter*x/(epsilon*(rotor_diameter*epsilon+k*x))
    # deltay = (0.25)*np.power(rotor_diameter, 2)*xi*(x-2.)/((rotor_diameter*epsilon+2*k)*(rotor_diameter*epsilon+k*x))
    # deltay = -(0.25)*rotor_diameter*xi*(2.*rotor_diameter-x)/((epsilon+2*k)*(rotor_diameter*epsilon+k*x))

    deltay = -(1./4.)*xi*rotor_diameter*(m*rotor_diameter-x)/(np.power(n, 2)*(epsilon+m*k)*(epsilon*rotor_diameter+k*x))

    return deltay #*np.cos(yaw)

if __name__ == "__main__":


    yaw = 20.0*np.pi/180.

    Rd = 126.4/2.0

    a = 1./3.

    Ct = 4.*a*(1.-a)

    wake_spread = 7.*np.pi/180.

    # for power
    m = 0.33
    m = 0.3
    n = -0.57
    nu = 0.56
    w = 1.3

    # for Bastankhah
    k = 0.05

    x = np.linspace(0.1, 15.*2.*Rd, 1000)

    y_approx = np.zeros_like(x)
    y_exact = np.zeros_like(x)
    y_exact_b = np.zeros_like(x)
    y_floris = np.zeros_like(x)
    y_power = np.zeros_like(x)
    y_bastankhah = np.zeros_like(x)

    for idx, x_val in enumerate(x):
        y_approx[idx] = -approx(x_val, wake_spread, yaw, Ct, Rd)
        y_exact[idx] = -exact(x_val, wake_spread, yaw, Ct, Rd)
        y_exact_b[idx] = -exact(x_val, wake_spread, yaw, Ct, Rd, a, True)
        y_floris[idx] = -exact(x_val, wake_spread, yaw, Ct, Rd)
        y_power[idx] = -ypower(x_val, m, n, nu, w, yaw, Ct, Rd)
        y_bastankhah[idx] = -bastankhah(x_val, k, yaw, Ct, Rd)

    plt.figure()
    # plt.plot(x/(2.*Rd), y_approx/(2.*Rd))
    plt.plot(x/(2.*Rd), y_floris/(2.*Rd), ':', label='FLORIS')
    plt.plot(x/(2.*Rd), y_exact/(2.*Rd), '--', label='Jensen')
    plt.plot(x/(2.*Rd), y_exact_b/(2.*Rd), 'c', label='Burton')
    plt.plot(x/(2.*Rd), y_power/(2.*Rd), 'r', label='Aitken')
    plt.plot(x/(2.*Rd), y_bastankhah/(2.*Rd), 'g', label='Bastankhah')
    print yaw
    plt.title('Yaw Offset for %.0f Degrees (m)' % (yaw*180.0/np.pi))
    plt.xlabel('Downstream Distance (x/D)')
    plt.ylabel('Offset Distance (y/D)')
    plt.legend()
    plt.show()