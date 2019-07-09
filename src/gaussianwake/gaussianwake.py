import numpy as np
from scipy.integrate import quad
from scipy.io import loadmat
import pylab as plt
import time

from openmdao.api import Component, Problem, Group

from _porteagel_fortran import porteagel_analyze as porteagel_analyze_fortran
from _porteagel_fortran import porteagel_analyze_bv, porteagel_analyze_dv
from _porteagel_fortran import theta_c_0_func, x0_func, sigmay_func, sigmaz_func, wake_offset_func

# def porteagel_analyze_fortran(turbineXw, turbineYw, turbineZ, rotorDiameter,
#                         Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I):
#
#     velocitiesTurbines = _porteagel_analyze(turbineXw, turbineYw, turbineZ, rotorDiameter,
#                         Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I)
#     # print velocitiesTurbines, 'here'
#     return velocitiesTurbines

def full_wake_offset_func(turbineXw, position_x, rotorDiameter, Ct, yaw, ky, kz, alpha, beta, I):
    #yaw = yaw * np.pi / 180.   # use if yaw is passed in as degrees
    x0 = x0_func(rotorDiameter, yaw, Ct, alpha, I, beta)

    theta_c_0 = theta_c_0_func(yaw, Ct)

    x = position_x - turbineXw

    sigmay = sigmay_func(x, x0, ky, rotorDiameter, yaw)

    sigmaz = sigmaz_func(x, x0, kz, rotorDiameter)

    wake_offset =  wake_offset_func(x, rotorDiameter, theta_c_0, x0, yaw, ky, kz, Ct, sigmay, sigmaz)

    return wake_offset

# def porteagel_analyze(nTurbines, turbineXw, turbineYw, turbineZ, rotorDiameter,
#                         Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I):
#
#     for i in range(0, nTurbines):
#         if (Ct[i] > 0.96):  # Glauert condition
#             axialInduction[i] = 0.143 + np.sqrt(0.0203 - 0.6427 * (0.889 - Ct[i]))
#         else:
#             axialInduction[i] = 0.5 * (1.0 - np.sqrt(1.0 - Ct[i]))
#
#     # NOTE: Bastankhah and Porte Agel 2016 defines yaw as positive clockwise, the negative below accounts for this
#     yaw *= -np.pi / 180.
#
#     velocitiesTurbines = np.tile(wind_speed, nTurbines)
#
#     for turb in range(0, nTurbines):
#         x0 = rotorDiameter[turb] * (np.cos(yaw[turb]) * (1.0 + np.sqrt(1.0 - Ct[turb])) /
#                                     (np.sqrt(2.0) * (alpha * I + beta * (1.0 - np.sqrt(1.0 - Ct[turb])))))
#         theta_c_0 = 0.3 * yaw[turb] * (1.0 - np.sqrt(1.0 - Ct[turb] * np.cos(yaw[turb]))) / np.cos(yaw[turb])
#
#         for turbI in range(0, nTurbines):  # at turbineX-locations
#
#             deltax0 = turbineXw[turbI] - (turbineXw[turb] + x0)
#
#             if deltax0 > 0.0:
#                 sigmay = rotorDiameter[turb] * (ky * deltax0 / rotorDiameter[turb]
#                                                 + np.cos(yaw[turb]) / np.sqrt(8.0))
#                 sigmaz = rotorDiameter[turb] * (kz * deltax0 / rotorDiameter[turb]
#                                                 + 1.0 / np.sqrt(8.0))
#                 wake_offset = rotorDiameter[turb] * (
#                     theta_c_0 * x0 / rotorDiameter[turb] +
#                     (theta_c_0 / 14.7) * np.sqrt(np.cos(yaw[turb]) / (ky * kz * Ct[turb])) *
#                     (2.9 + 1.3 * np.sqrt(1.0 - Ct[turb]) - Ct[turb]) *
#                     np.log(
#                         ((1.6 + np.sqrt(Ct[turb])) *
#                          (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
#                                         (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
#                           - np.sqrt(Ct[turb]))) /
#                         ((1.6 - np.sqrt(Ct[turb])) *
#                          (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
#                                         (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
#                           + np.sqrt(Ct[turb])))
#                     )
#                 )
#                 # print wake_offset, turbineYw[turbI]
#
#                 deltay = turbineYw[turbI] - (turbineYw[turb] + wake_offset)
#
#                 deltav = wind_speed * (
#                     (1.0 - np.sqrt(1.0 - Ct[turb] *
#                                    np.cos(yaw[turb]) / (8.0 * sigmay * sigmaz /
#                                                         (rotorDiameter[turb] ** 2)))) *
#                     np.exp(-0.5 * ((deltay) / sigmay) ** 2) *
#                     np.exp(-0.5 * ((turbineZ[turbI] - turbineZ[turb]) / sigmaz) ** 2)
#                 )
#
#                 velocitiesTurbines[turbI] -= deltav
#
#     return velocitiesTurbines
#
#
# def porteagel_visualize(nTurbines, nSamples, turbineXw, turbineYw, turbineZ, velX, velY, velZ, rotorDiameter,
#                         Ct, axialInduction, wind_speed, yaw, ky, kz, alpha, beta, I):
#
#     for i in range(0, nTurbines):
#         if (Ct[i] > 0.96):  # Glauert condition
#             axialInduction[i] = 0.143 + np.sqrt(0.0203 - 0.6427 * (0.889 - Ct[i]))
#         else:
#             axialInduction[i] = 0.5 * (1.0 - np.sqrt(1.0 - Ct[i]))
#
#     # NOTE: Bastankhah and Porte Agel 2016 defines yaw as positive clockwise, the negative below accounts for this
#     yaw *= -np.pi / 180.
#     ws_array = np.tile(wind_speed, nSamples)
#
#     for turb in range(0, nTurbines):
#         x0 = rotorDiameter[turb] * (np.cos(yaw[turb]) * (1.0 + np.sqrt(1.0 - Ct[turb])) /
#                                     (np.sqrt(2.0) * (alpha * I + beta * (1.0 - np.sqrt(1.0 - Ct[turb])))))
#         theta_c_0 = 0.3 * yaw[turb] * (1.0 - np.sqrt(1.0 - Ct[turb] * np.cos(yaw[turb]))) / np.cos(yaw[turb])
#
#         for loc in range(0, nSamples):  # at velX-locations
#             deltax0 = velX[loc] - (turbineXw[turb] + x0)
#             if deltax0 > 0.0:
#                 sigmay = rotorDiameter[turb] * (ky * deltax0 / rotorDiameter[turb]
#                                                 + np.cos(yaw[turb]) / np.sqrt(8.0))
#                 sigmaz = rotorDiameter[turb] * (kz * deltax0 / rotorDiameter[turb]
#                                                 + 1.0 / np.sqrt(8.0))
#                 wake_offset = rotorDiameter[turb] * (
#                     theta_c_0 * x0 / rotorDiameter[turb] +
#                     (theta_c_0 / 14.7) * np.sqrt(np.cos(yaw[turb]) / (ky * kz * Ct[turb])) *
#                     (2.9 + 1.3 * np.sqrt(1.0 - Ct[turb]) - Ct[turb]) *
#                     np.log(
#                         ((1.6 + np.sqrt(Ct[turb])) *
#                          (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
#                                         (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
#                           - np.sqrt(Ct[turb]))) /
#                         ((1.6 - np.sqrt(Ct[turb])) *
#                          (1.6 * np.sqrt(8.0 * sigmay * sigmaz /
#                                         (np.cos(yaw[turb]) * rotorDiameter[turb] ** 2))
#                           + np.sqrt(Ct[turb])))
#                     )
#                 )
#
#                 deltay = velY[loc] - (turbineYw[turb] + wake_offset)
#
#                 deltav = wind_speed * (
#                     (1.0 - np.sqrt(1.0 - Ct[turb] *
#                                    np.cos(yaw[turb]) / (8.0 * sigmay * sigmaz /
#                                                         (rotorDiameter[turb] ** 2)))) *
#                     np.exp(-0.5 * ((deltay) / sigmay) ** 2) *
#                     np.exp(-0.5 * ((velZ[loc] - turbineZ[turb]) / sigmaz) ** 2)
#                 )
#
#                 ws_array[loc] -= deltav
#
#     return ws_array


class GaussianWake(Component):

    def __init__(self, nTurbines, direction_id=0, options=None):
        super(GaussianWake, self).__init__()

        import warnings

        self.deriv_options['type'] = 'user'
        # self.deriv_options['form'] = 'central'
        # self.deriv_options['step_size'] = 1.0e-12
        # self.deriv_options['step_calc'] = 'relative'

        self.nTurbines = nTurbines
        self.direction_id = direction_id

        if options is None:
            self.radius_multiplier = 1.0
            self.nSamples = nSamples = 0
            self.nRotorPoints = 1
            self.use_ct_curve = False
            self.ct_curve_ct = np.array([0.0])
            self.ct_curve_wind_speed = np.array([0.0])
            self.interp_type = 1
        else:
            # self.radius_multiplier = options['radius multiplier']
            try:
                self.nSamples = nSamples = options['nSamples']
            except:
                self.nSamples = nSamples = 0
            try:
                self.nRotorPoints = nRotorPoints = options['nRotorPoints']
            except:
                self.nRotorPoints = nRotorPoints = 1
            try:
                self.use_ct_curve = options['use_ct_curve']
                self.ct_curve_ct = options['ct_curve_ct']
                self.ct_curve_wind_speed = options['ct_curve_wind_speed']

            except:
                self.use_ct_curve = False
                self.ct_curve_ct = np.array([0.0])
                self.ct_curve_wind_speed = np.array([0.0])
            try:
                self.interp_type = options['interp_type']
            except:
                self.interp_type = 1

        ct_max = 0.99
        if np.any(self.ct_curve_ct > 0.):
            if np.any(self.ct_curve_ct > ct_max):
                warnings.warn('Ct values must be <= 1, clipping provided values accordingly')
                self.ct_curve_ct = np.clip(self.ct_curve_ct, a_max=ct_max, a_min=None)

        # unused but required for compatibility

        # self.add_param('wakeCentersYT', np.zeros(nTurbines*nTurbines), units='m')
        # self.add_param('wakeDiametersT', np.zeros(nTurbines*nTurbines), units='m')
        # self.add_param('wakeOverlapTRel', np.zeros(nTurbines*nTurbines))

        # used
        self.add_param('turbineXw', val=np.zeros(nTurbines), units='m')
        self.add_param('turbineYw', val=np.zeros(nTurbines), units='m')
        self.add_param('hubHeight', val=np.ones(nTurbines)*90.0, units='m')
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
        self.add_param('model_params:I', val=0.075, pass_by_object=True, desc='turbulence intensity')
        self.add_param('model_params:z_ref', val=80.0, pass_by_object=True, desc='wind speed measurement height')
        self.add_param('model_params:z_0', val=0.0, pass_by_object=True, desc='ground height')
        self.add_param('model_params:shear_exp', val=0.15, pass_by_object=True, desc='wind shear calculation exponent')
        self.add_param('model_params:wake_combination_method', val=1, pass_by_object=True,
                       desc='select how the wakes should be combined')
        self.add_param('model_params:ti_calculation_method', val=2, pass_by_object=True,
                       desc='select how the wakes should be combined')
        self.add_param('model_params:calc_k_star', val=True, pass_by_object=True,
                       desc='choose to calculate wake expansion based on TI if True')
        self.add_param('model_params:sort', val=True, pass_by_object=True,
                       desc='decide whether turbines should be sorted before solving for directional power')
        self.add_param('model_params:RotorPointsY', val=np.zeros(nRotorPoints), pass_by_object=True,
                       desc='rotor swept area sampling Y points centered at (y,z)=(0,0) normalized by rotor radius')
        self.add_param('model_params:RotorPointsZ', val=np.zeros(nRotorPoints), pass_by_object=True,
                       desc='rotor swept area sampling Z points centered at (y,z)=(0,0) normalized by rotor radius')
        self.add_param('model_params:print_ti', val=False, pass_by_object=True,
                       desc='print TI values to a file for use in plotting etc')
        self.add_param('model_params:wake_model_version', val=2016, pass_by_object=True,
                       desc='choose whether to use Bastankhah 2014 or 2016')

        self.add_param('model_params:wec_factor', val=1.0, pass_by_object=True,
                       desc='increase spread for optimization')

        self.add_param('model_params:sm_smoothing', val=700.0, pass_by_object=True,
                       desc='adjust degree of smoothing in the smooth-max for local TI calcs')

        self.add_param('model_params:exp_rate_multiplier', val=1.0, pass_by_object=True,
                       desc='multiply wake expansion rate as wec alternative')

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
        wake_combination_method = params['model_params:wake_combination_method']
        ti_calculation_method = params['model_params:ti_calculation_method']
        calc_k_star = params['model_params:calc_k_star']
        sort_turbs = params['model_params:sort']
        RotorPointsY = params['model_params:RotorPointsY']
        RotorPointsZ = params['model_params:RotorPointsZ']
        z_ref = params['model_params:z_ref']
        z_0 = params['model_params:z_0']
        shear_exp = params['model_params:shear_exp']
        wake_model_version = params['model_params:wake_model_version']

        wec_factor = params['model_params:wec_factor']

        print_ti = params['model_params:print_ti']

        sm_smoothing = params['model_params:sm_smoothing']
        exp_rate_multiplier = params['model_params:exp_rate_multiplier']


        # rename inputs and outputs
        turbineXw = params['turbineXw']
        turbineYw = params['turbineYw']
        turbineZ = params['hubHeight']
        yaw = params['yaw%i' % direction_id]
        rotorDiameter = params['rotorDiameter']
        Ct = params['Ct']
        wind_speed = params['wind_speed']

        use_ct_curve = self.use_ct_curve
        interp_type = self.interp_type

        if use_ct_curve:
            ct_curve_wind_speed = self.ct_curve_wind_speed
            ct_curve_ct = self.ct_curve_ct
        else:
            ct_curve_wind_speed = np.ones_like(Ct)*wind_speed
            ct_curve_ct = Ct

        # run the Bastankhah and Porte Agel model
        # velocitiesTurbines = porteagel_analyze(nTurbines=nTurbines, turbineXw=turbineXw, turbineYw=turbineYw,
        #                                        turbineZ=turbineZ, rotorDiameter=rotorDiameter, Ct=Ct,
        #                                        axialInduction=axialInduction, wind_speed=wind_speed, yaw=np.copy(yaw),
        #                                        ky=ky, kz=kz, alpha=alpha, beta=beta, I=I)
        if sort_turbs:
            sorted_x_idx = np.argsort(turbineXw, kind='heapsort')
        else:
            sorted_x_idx = np.arange(0, nTurbines)

        self.sorted_x_idx = sorted_x_idx

        if nSamples > 0:
            CalculateFlowField = True
            FieldPointsX = params['wsPositionXw']
            FieldPointsY = params['wsPositionYw']
            FieldPointsZ = params['wsPositionZ']

        else:
            CalculateFlowField = False
            FieldPointsX = np.array([0])
            FieldPointsY = np.array([0])
            FieldPointsZ = np.array([0])

        TurbineVelocity, FieldVelocity = porteagel_analyze_fortran(turbineXw, sorted_x_idx, turbineYw,
                                                       turbineZ, rotorDiameter, Ct,
                                                       wind_speed, np.copy(yaw),
                                                       ky, kz, alpha, beta, I, RotorPointsY, RotorPointsZ, FieldPointsX,
                                                       FieldPointsY, FieldPointsZ, z_ref, z_0, shear_exp,
                                                       wake_combination_method, ti_calculation_method,
                                                       calc_k_star, wec_factor, print_ti, wake_model_version,
                                                       interp_type, use_ct_curve, ct_curve_wind_speed, ct_curve_ct,
                                                       sm_smoothing, exp_rate_multiplier, CalculateFlowField)

        unknowns['wtVelocity%i' % direction_id] = TurbineVelocity

        if nSamples > 0:
            unknowns['wsArray%i' % direction_id] = FieldVelocity

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

        wake_combination_method = params['model_params:wake_combination_method']
        ti_calculation_method = params['model_params:ti_calculation_method']
        calc_k_star = params['model_params:calc_k_star']
        sort_turbs = params['model_params:sort']
        RotorPointsY = params['model_params:RotorPointsY']
        RotorPointsZ = params['model_params:RotorPointsZ']
        z_ref = params['model_params:z_ref']
        z_0 = params['model_params:z_0']
        shear_exp = params['model_params:shear_exp']

        wec_factor = params['model_params:wec_factor']

        wake_model_version = params['model_params:wake_model_version']

        sm_smoothing = params['model_params:sm_smoothing']

        exp_rate_multiplier = params['model_params:exp_rate_multiplier']

        use_ct_curve = self.use_ct_curve
        interp_type = self.interp_type

        if use_ct_curve:
            ct_curve_wind_speed = self.ct_curve_wind_speed
            ct_curve_ct = self.ct_curve_ct
        else:
            ct_curve_wind_speed = np.ones_like(Ct)*wind_speed
            ct_curve_ct = Ct

        print_ti = False

        sorted_x_idx = self.sorted_x_idx

        # define jacobian size
        nTurbines = len(turbineXw)
        nDirs = nTurbines



        # print("before calling gradients")
        # call to fortran code to obtain output values
        # turbineXwb, turbineYwb, turbineZb, rotorDiameterb, Ctb, yawDegb =
            # porteagel_analyze_bv(turbineXw, sorted_x_idx, turbineYw, turbineZ,
            #                      rotorDiameter, Ct, wind_speed, yawDeg,
            #                      ky, kz, alpha, beta, I, RotorPointsY, RotorPointsZ,z_ref, z_0,
            #                      shear_exp, wake_combination_method, ti_calculation_method, calc_k_star,
            #                      wec_factor, print_ti, wake_model_version, interp_type, use_ct_curve, ct_curve_wind_speed,
            #                      ct_curve_ct, wtVelocityb)

        # define input array to direct differentiation
        wtVelocityb = np.eye(nDirs, nTurbines)
        FieldPointsX = np.array([0])
        FieldPointsY = np.array([0])
        FieldPointsZ = np.array([0])
        CalculateFlowField = False


        turbineXwb, turbineYwb, turbineZb, rotorDiameter, Ctb, yawDegb = porteagel_analyze_bv(turbineXw, sorted_x_idx,
                                                                                               turbineYw, turbineZ,
                                                                                               rotorDiameter, Ct,
                                                                                               wind_speed, yawDeg, ky,
                                                                                               kz, alpha, beta, I,
                                                                                               RotorPointsY,
                                                                                               RotorPointsZ,
                                                                                               FieldPointsX,
                                                                                               FieldPointsY,
                                                                                               FieldPointsZ, z_ref, z_0,
                                                                                               shear_exp,
                                                                                               wake_combination_method,
                                                                                               ti_calculation_method,
                                                                                               calc_k_star, wec_factor,
                                                                                               print_ti,
                                                                                               wake_model_version,
                                                                                               interp_type,
                                                                                               use_ct_curve,
                                                                                               ct_curve_wind_speed,
                                                                                               ct_curve_ct,
                                                                                               sm_smoothing,
                                                                                               exp_rate_multiplier,
                                                                                               CalculateFlowField,
                                                                                               wtVelocityb)
        #
        # turbineXwd = np.eye(nDirs, nTurbines)
        # turbineYwd = np.zeros([nDirs, nTurbines])
        # turbineZd = np.zeros([nDirs, nTurbines])
        # rotorDiameterd = np.zeros([nDirs, nTurbines])
        # Ctd = np.zeros([nDirs, nTurbines])
        # yawDegd = np.zeros([nDirs, nTurbines])
        # FieldPointsX = np.array([0])
        # FieldPointsY = np.array([0])
        # FieldPointsZ = np.array([0])
        # CalculateFlowField = False

        # _, wtVelocityd, _ = porteagel_analyze_dv(turbineXw, turbineXwd, sorted_x_idx, turbineYw, turbineYwd, turbineZ, turbineZd,
        #                      rotorDiameter, rotorDiameterd, Ct, Ctd, wind_speed, yawDeg, yawDegd, ky, kz, alpha, beta,
        #                      I, RotorPointsY, RotorPointsZ, FieldPointsX, FieldPointsY, FieldPointsZ, z_ref, z_0, shear_exp, wake_combination_method,
        #                      ti_calculation_method, calc_k_star, wec_factor, print_ti, wake_model_version, interp_type,
        #                      use_ct_curve, ct_curve_wind_speed, ct_curve_ct, sm_smoothing, exp_rate_multiplier, CalculateFlowField)
        #
        # wtVelocityb_dxwd = wtVelocityd
        #
        # turbineXwd = np.zeros([nDirs, nTurbines])
        # turbineYwd = np.eye(nDirs, nTurbines)
        # turbineZd = np.zeros([nDirs, nTurbines])
        # rotorDiameterd = np.zeros([nDirs, nTurbines])
        # Ctd = np.zeros([nDirs, nTurbines])
        # yawDegd = np.zeros([nDirs, nTurbines])
        #
        # t0, wtVelocityd, t2 = porteagel_analyze_dv(turbineXw, turbineXwd, sorted_x_idx, turbineYw, turbineYwd, turbineZ,
        #                                          turbineZd,
        #                                          rotorDiameter, rotorDiameterd, Ct, Ctd, wind_speed, yawDeg, yawDegd,
        #                                          ky, kz, alpha, beta,
        #                                          I, RotorPointsY, RotorPointsZ, FieldPointsX, FieldPointsY,
        #                                          FieldPointsZ, z_ref, z_0, shear_exp, wake_combination_method,
        #                                          ti_calculation_method, calc_k_star, wec_factor, print_ti,
        #                                          wake_model_version, interp_type,
        #                                          use_ct_curve, ct_curve_wind_speed, ct_curve_ct,
        #                                          sm_smoothing, exp_rate_multiplier, CalculateFlowField)
        #
        # wtVelocityb_dywd = wtVelocityd
        #
        # turbineXwd = np.zeros([nDirs, nTurbines])
        # turbineYwd = np.zeros([nDirs, nTurbines])
        # turbineZd = np.eye(nDirs, nTurbines)
        # rotorDiameterd = np.zeros([nDirs, nTurbines])
        # Ctd = np.zeros([nDirs, nTurbines])
        # yawDegd = np.zeros([nDirs, nTurbines])
        #
        # _, wtVelocityd, _ = porteagel_analyze_dv(turbineXw, turbineXwd, sorted_x_idx, turbineYw, turbineYwd, turbineZ,
        #                                          turbineZd,
        #                                          rotorDiameter, rotorDiameterd, Ct, Ctd, wind_speed, yawDeg, yawDegd,
        #                                          ky, kz, alpha, beta,
        #                                          I, RotorPointsY, RotorPointsZ, FieldPointsX, FieldPointsY,
        #                                          FieldPointsZ, z_ref, z_0, shear_exp, wake_combination_method,
        #                                          ti_calculation_method, calc_k_star, wec_factor, print_ti,
        #                                          wake_model_version, interp_type,
        #                                          use_ct_curve, ct_curve_wind_speed, ct_curve_ct,
        #                                          sm_smoothing, exp_rate_multiplier, CalculateFlowField)
        #
        # wtVelocityb_dzd = wtVelocityd
        #
        # turbineXwd = np.zeros([nDirs, nTurbines])
        # turbineYwd = np.zeros([nDirs, nTurbines])
        # turbineZd = np.zeros([nDirs, nTurbines])
        # rotorDiameterd = np.eye(nDirs, nTurbines)
        # Ctd = np.zeros([nDirs, nTurbines])
        # yawDegd = np.zeros([nDirs, nTurbines])
        #
        # _, wtVelocityd, _ = porteagel_analyze_dv(turbineXw, turbineXwd, sorted_x_idx, turbineYw, turbineYwd, turbineZ,
        #                                          turbineZd,
        #                                          rotorDiameter, rotorDiameterd, Ct, Ctd, wind_speed, yawDeg, yawDegd,
        #                                          ky, kz, alpha, beta,
        #                                          I, RotorPointsY, RotorPointsZ, FieldPointsX, FieldPointsY,
        #                                          FieldPointsZ, z_ref, z_0, shear_exp, wake_combination_method,
        #                                          ti_calculation_method, calc_k_star, wec_factor, print_ti,
        #                                          wake_model_version, interp_type,
        #                                          use_ct_curve, ct_curve_wind_speed, ct_curve_ct,
        #                                          sm_smoothing, exp_rate_multiplier, CalculateFlowField)
        #
        # wtVelocityb_drd = wtVelocityd
        #
        # turbineXwd = np.zeros([nDirs, nTurbines])
        # turbineYwd = np.zeros([nDirs, nTurbines])
        # turbineZd = np.zeros([nDirs, nTurbines])
        # rotorDiameterd = np.zeros([nDirs, nTurbines])
        # Ctd = np.eye(nDirs, nTurbines)
        # yawDegd = np.zeros([nDirs, nTurbines])
        #
        # _, wtVelocityd, _ = porteagel_analyze_dv(turbineXw, turbineXwd, sorted_x_idx, turbineYw, turbineYwd, turbineZ,
        #                                          turbineZd,
        #                                          rotorDiameter, rotorDiameterd, Ct, Ctd, wind_speed, yawDeg, yawDegd,
        #                                          ky, kz, alpha, beta,
        #                                          I, RotorPointsY, RotorPointsZ, FieldPointsX, FieldPointsY,
        #                                          FieldPointsZ, z_ref, z_0, shear_exp, wake_combination_method,
        #                                          ti_calculation_method, calc_k_star, wec_factor, print_ti,
        #                                          wake_model_version, interp_type,
        #                                          use_ct_curve, ct_curve_wind_speed, ct_curve_ct,
        #                                          sm_smoothing, exp_rate_multiplier, CalculateFlowField)
        #
        # wtVelocityb_dctd = wtVelocityd
        #
        # turbineXwd = np.zeros([nDirs, nTurbines])
        # turbineYwd = np.zeros([nDirs, nTurbines])
        # turbineZd = np.zeros([nDirs, nTurbines])
        # rotorDiameterd = np.zeros([nDirs, nTurbines])
        # Ctd = np.zeros([nDirs, nTurbines])
        # yawDegd = np.eye(nDirs, nTurbines)
        #
        # _, wtVelocityd, _ = porteagel_analyze_dv(turbineXw, turbineXwd, sorted_x_idx, turbineYw, turbineYwd, turbineZ,
        #                                          turbineZd,
        #                                          rotorDiameter, rotorDiameterd, Ct, Ctd, wind_speed, yawDeg, yawDegd,
        #                                          ky, kz, alpha, beta,
        #                                          I, RotorPointsY, RotorPointsZ, FieldPointsX, FieldPointsY,
        #                                          FieldPointsZ, z_ref, z_0, shear_exp, wake_combination_method,
        #                                          ti_calculation_method, calc_k_star, wec_factor, print_ti,
        #                                          wake_model_version, interp_type,
        #                                          use_ct_curve, ct_curve_wind_speed, ct_curve_ct,
        #                                          sm_smoothing, exp_rate_multiplier, CalculateFlowField)
        #
        # wtVelocityb_dyawd = wtVelocityd

        # print(wtVelocityb_dxwd, wtVelocityb[1])
        # print(wtVelocityb_dxwd.shape, wtVelocityb.shape)
        # for i in np.arange(0, nDirs):
        #     print(wtVelocityb_dxwd, wtVelocityb)
        #     print(wtVelocityb_dxwd.shape, wtVelocityb.shape)
        #     wtVelocityb_dxwd[:, i] = wtVelocityb[i]

        # print("after calling gradients")

        # quit()
        # print(wtVelocityb.shape)

        # print(wtVelocityb)

        # quit()

        # initialize Jacobian dict
        J = {}

        # # collect values of the Jacobian
        # J['wtVelocity%i' % direction_id, 'turbineXw'] = wtVelocityb[0, :]
        # J['wtVelocity%i' % direction_id, 'turbineYw'] = wtVelocityb[1, :]
        # J['wtVelocity%i' % direction_id, 'hubHeight'] = wtVelocityb[2, :]
        # J['wtVelocity%i' % direction_id, 'yaw%i' % direction_id] = wtVelocityb[3, :]
        # J['wtVelocity%i' % direction_id, 'rotorDiameter'] = wtVelocityb[4, :]
        # J['wtVelocity%i' % direction_id, 'Ct'] = wtVelocityb[5, :]
        # turbineXwb, turbineYwb, turbineZb, rotorDiameter, Ctb, yawDegb
        J['wtVelocity%i' % direction_id, 'turbineXw'] = turbineXwb
        J['wtVelocity%i' % direction_id, 'turbineYw'] = turbineYwb
        J['wtVelocity%i' % direction_id, 'hubHeight'] = turbineZb
        J['wtVelocity%i' % direction_id, 'yaw%i' % direction_id] = yawDegb
        J['wtVelocity%i' % direction_id, 'rotorDiameter'] = rotorDiameter
        J['wtVelocity%i' % direction_id, 'Ct'] = Ctb
        # print J

        # print("shapes")
        # print(wtVelocityb)
        # print(wtVelocityb_dxwd)

        # J['wtVelocity%i' % direction_id, 'turbineXw'] = np.transpose(wtVelocityb_dxwd)
        # J['wtVelocity%i' % direction_id, 'turbineYw'] = np.transpose(wtVelocityb_dywd)
        # J['wtVelocity%i' % direction_id, 'hubHeight'] = np.transpose(wtVelocityb_dzd)
        # J['wtVelocity%i' % direction_id, 'yaw%i' % direction_id] = np.transpose(wtVelocityb_dyawd)
        # J['wtVelocity%i' % direction_id, 'rotorDiameter'] = np.transpose(wtVelocityb_drd)
        # J['wtVelocity%i' % direction_id, 'Ct'] = np.transpose(wtVelocityb_dctd)
        # # print J

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