! Implementation of the Bastankhah and Porte Agel gaussian-shaped wind turbine wake 
! model (2016) with various farm modeling (TI and wake combination) methods included
! Created by Jared J. Thomas, 2017-2019.
! FLight Optimization and Wind Laboratory (FLOW Lab)
! Brigham Young University

! implementation of the Bastankhah and Porte Agel (BPA) wake model for analysis
subroutine porteagel_analyze(nTurbines, nRotorPoints, nCtPoints, nFieldPoints, turbineXw, &
                             sorted_x_idx, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, TI, RotorPointsY, RotorPointsZ, &
                             FieldPointsX, FieldPointsY, FieldPointsZ, &
                             z_ref, z_0, shear_exp, wake_combination_method, &
                             TI_calculation_method, calc_k_star, wec_factor, print_ti, &
                             wake_model_version, interp_type, &
                             use_ct_curve, ct_curve_wind_speed, ct_curve_ct, sm_smoothing, &
                             wec_spreading_angle, CalculateFlowField, WECH, &
                             wtVelocity, FieldVelocity)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter Ct yawDeg

    ! dependent variables:


    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nRotorPoints, nCtPoints, nFieldPoints
    integer, intent(in) :: wake_combination_method, TI_calculation_method, & 
                        &  wake_model_version, interp_type, WECH
    logical, intent(in) :: calc_k_star, print_ti, use_ct_curve, CalculateFlowField
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    integer, dimension(nTurbines), intent(in) :: sorted_x_idx
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, TI, wind_speed, z_ref, z_0, shear_exp, wec_factor
    real(dp), dimension(nRotorPoints), intent(in) :: RotorPointsY, RotorPointsZ
    real(dp), dimension(nCtPoints), intent(in) :: ct_curve_wind_speed, ct_curve_ct
    real(dp), intent(in) :: sm_smoothing, wec_spreading_angle
    real(dp), dimension(nFieldPoints), intent(in) :: FieldPointsX, FieldPointsY, FieldPointsZ

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw, TIturbs, Ct_local, ky_local, kz_local
    real(dp) :: k_star
    real(dp) :: tol, TI_area_ratio 
    real(dp) :: TI_area_ratio_tmp, TI_dst_tmp, TI_ust_tmp, rpts
    real(dp) :: LocalRotorPointY, LocalRotorPointZ
    real(dp) :: pointX, pointY, pointZ, point_velocity_with_shear
    real(dp) :: x, x0, theta_c_0, deltay, deltax0, sigmay, sigmaz, wake_offset
    Integer :: d, u, turbI, p, turb
    real(dp), parameter :: pi = 3.141592653589793_dp
    
    ! model out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity
    real(dp), dimension(nFieldPoints), intent(out) :: FieldVelocity

    ! initialize intrinsic functions
    intrinsic sin, cos, atan, max, sqrt, log
    
    ! bastankhah and porte agel 2016 defines yaw to be positive clockwise, this is 
    ! reversed from the convention used in plant energy and from typical convention
    yaw = - yawDeg*pi/180.0_dp
    
    ! set tolerance for location checks
    tol = 0.1_dp
    
    ! initialize wind turbine velocities to 0.0
    wtVelocity = 0.0_dp
    
    ! initialize local TI of all turbines to free-stream value
    TIturbs(:) = TI
    
    ! initialize the local wake factors
    if (calc_k_star .eqv. .true.) then
        call k_star_func(TI, k_star)
        ky_local(:) = k_star
        kz_local(:) = k_star
    else
        ky_local(:) = ky
        kz_local(:) = kz
    end if
    
    Ct_local(:) = Ct

    do, d=1, nTurbines
    
        ! get index of downstream turbine
        turbI = sorted_x_idx(d) + 1
        
        do, p=1, nRotorPoints
        
            ! scale rotor sample point coordinate by rotor diameter (in rotor hub ref. frame)
            LocalRotorPointY = RotorPointsY(p)*0.5_dp*rotorDiameter(turbI)
            LocalRotorPointZ = RotorPointsZ(p)*0.5_dp*rotorDiameter(turbI)
            pointX = turbineXw(turbI) + LocalRotorPointY*sin(yaw(turbI)) 
            pointY = turbineYw(turbI) + LocalRotorPointY*cos(yaw(turbI)) 
            pointZ = turbineZ(turbI) + LocalRotorPointZ
            
            ! calculate the velocity at given point
            call point_velocity_with_shear_func(nTurbines, turbI, wake_combination_method, &
                                          wake_model_version, &
                                          sorted_x_idx, pointX, pointY, pointZ, &
                                          tol, alpha, beta, wec_spreading_angle, wec_factor, &
                                          wind_speed, z_ref, z_0, shear_exp, &
                                          turbineXw, turbineYw, turbineZ, &
                                          rotorDiameter, yaw, wtVelocity, &
                                          Ct_local, TIturbs, ky_local, kz_local, WECH, &
                                          point_velocity_with_shear)
            
            ! add sample point velocity to turbine velocity to be averaged later
            wtVelocity(turbI) = wtVelocity(turbI) + point_velocity_with_shear
        
        end do
    
        ! final velocity calculation for turbine turbI (average equally across all points)
        rpts = REAL(nRotorPoints, dp)

        wtVelocity(turbI) = wtVelocity(turbI)/rpts
        
        ! update thrust coefficient for turbI
        if (use_ct_curve) then
            call interpolation(nCtPoints, interp_type, ct_curve_wind_speed, ct_curve_ct, & 
                              & wtVelocity(turbI), Ct_local(turbI), 0.0_dp, 0.0_dp, .false.)
        end if
        
        ! calculate local turbulence intensity at turbI
        if (TI_calculation_method > 0) then
        
            ! initialize the TI_area_ratio to 0.0 for each turbine
            TI_area_ratio = 0.0_dp
    
            ! initialize local ti tmp
            TI_dst_tmp = TIturbs(turbI)
            
            ! loop over upstream turbines
            do, u=1, nTurbines
            
                ! get index of upstream turbine
                turb = sorted_x_idx(u) + 1
                
                ! skip turbine's influence on itself
                if (turb .eq. turbI) cycle
                
                ! calculate downstream distance between wind turbines
                x = turbineXw(turbI) - turbineXw(turb)
                
                if (x > tol) then
                    ! determine the far-wake onset location 
                    call x0_func(rotorDiameter(turb), yaw(turb), Ct_local(turb), alpha, & 
                                & TIturbs(turb), beta, x0)
                
                    ! calculate the distance from the onset of far-wake
                    deltax0 = x - x0
                
                    ! horizontal spread 
                    call sigmay_func(x, x0, ky_local(turb), rotorDiameter(turb), yaw(turb), sigmay)

                    ! vertical spread 
                    call sigmaz_func(x, x0, kz_local(turb), rotorDiameter(turb), sigmaz)
                
                    ! determine the initial wake angle at the onset of far wake
                    call theta_c_0_func(yaw(turb), Ct_local(turb), theta_c_0)
            
                    ! horizontal cross-wind wake displacement from hub
                    call wake_offset_func(x, rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                         & ky_local(turb), kz_local(turb), &
                                         Ct_local(turb), sigmay, sigmaz, wake_offset)
            
                    ! cross wind distance from point location to upstream turbine wake center
                    deltay = turbineYw(turbI) - (turbineYw(turb) + wake_offset)   
                
                    ! save ti_area_ratio and ti_dst to new memory locations to avoid 
                    ! aliasing during differentiation
                    TI_area_ratio_tmp = TI_area_ratio
                    TI_dst_tmp = TIturbs(turbI)
                    TI_ust_tmp = TIturbs(turb)
            
                    ! update local turbulence intensity
                    call added_ti_func(TI, Ct_local(turb), x, ky_local(turb), rotorDiameter(turb), & 
                                       & rotorDiameter(turbI), deltay, turbineZ(turb), &
                                       & turbineZ(turbI), sm_smoothing, TI_ust_tmp, &
                                       & TI_calculation_method, TI_area_ratio_tmp, &
                                       & TI_dst_tmp, TI_area_ratio, TIturbs(turbI))
                end if
            
            end do
            
            ! calculate wake spreading parameter at turbI based on local turbulence intensity
            if (calc_k_star .eqv. .true.) then
        
                call k_star_func(TIturbs(turbI), k_star)
                ky_local(turbI) = k_star
                kz_local(turbI) = k_star
        
            end if
            
        end if

    end do
    
    ! calculate flow field
    if (CalculateFlowField) then
        do, p=1, nFieldPoints
            
            ! calculate the velocity at given point
            call point_velocity_with_shear_func(nTurbines, turbI, wake_combination_method, &
                                          wake_model_version, &
                                          sorted_x_idx, FieldPointsX(p), FieldPointsY(p), &
                                          FieldPointsZ(p), &
                                          tol, alpha, beta, wec_spreading_angle, wec_factor, &
                                          wind_speed, z_ref, z_0, shear_exp, &
                                          turbineXw, turbineYw, turbineZ, &
                                          rotorDiameter, yaw, wtVelocity, &
                                          Ct_local, TIturbs, ky_local, kz_local, WECH,&
                                          FieldVelocity(p))
        
        end do
    end if

end subroutine porteagel_analyze

subroutine point_velocity_with_shear_func(nTurbines, turbI, wake_combination_method, &
                                          wake_model_version, &
                                          sorted_x_idx, pointX, pointY, pointZ, &
                                          tol, alpha, beta, wec_spreading_angle, wec_factor, &
                                          wind_speed, z_ref, z_0, shear_exp, &
                                          turbineXw, turbineYw, turbineZ, &
                                          rotorDiameter, yaw, wtVelocity, &
                                          Ct_local, TIturbs, ky_local, kz_local, WECH, &
                                          point_velocity_with_shear)
                                          
    ! if not calculating velocity for a specific turbine, please set turbI to 0

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    Integer, intent(in) :: nTurbines, turbI, wake_combination_method, wake_model_version, WECH
    Integer, dimension(nTurbines), intent(in) :: sorted_x_idx
    Real(dp), intent(in) :: pointX, pointY, pointZ
    Real(dp), intent(in) :: tol, alpha, beta, wec_spreading_angle, wec_factor
    Real(dp), intent(in) :: wind_speed
    Real(dp), intent(in) :: z_ref, z_0, shear_exp
    Real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    Real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yaw, wtVelocity
    Real(dp), dimension(nTurbines), intent(in) :: Ct_local, TIturbs, ky_local, kz_local
    
    ! out
     Real(dp), intent(out) :: point_velocity_with_shear
     Real(dp) :: wake_offset
    
    ! local
    Real(dp) :: old_deficit_sum, deficit_sum
    Real(dp) :: x, deltav, x0, theta_c_0, sigmay, sigmaz
    Real(dp) :: discontinuity_point, sigmay_d, sigmaz_d
    Real(dp) :: sigmay_0, sigmaz_0, deltay, deltaz, point_velocity
    Real(dp) :: sigmay_spread, sigmaz_spread, sigmay_0_spread, sigmaz_0_spread
    Integer :: u, turb

    ! initialize deficit summation term to zero
    deficit_sum = 0.0_dp
    
    ! loop through all turbines
    do, u=1, nTurbines ! at turbineX-locations
        
        ! get index of upstream turbine
        turb = sorted_x_idx(u) + 1
        
        ! skip this loop if turb = turbI (turbines impact on itself)
        if (turb .eq. turbI) cycle
    
        ! downstream distance between upstream turbine and point
        x = pointX - turbineXw(turb)
    
        ! set this iterations velocity deficit to 0
        deltav = 0.0_dp
        
        ! check turbine relative locations
        if (x > tol) then
        
            ! determine the onset location of far wake
            call x0_func(rotorDiameter(turb), yaw(turb), Ct_local(turb), alpha, & 
                        & TIturbs(turb), beta, x0)
            
            ! find the final point where the original model is undefined
            call discontinuity_point_func(x0, rotorDiameter(turb), & 
                                         ky_local(turb), kz_local(turb), &
                                         yaw(turb), Ct_local(turb), & 
                                         discontinuity_point)  
                                         
            ! horizontal spread at discontinuity point
            call sigmay_func(discontinuity_point, x0, ky_local(turb), rotorDiameter(turb), yaw(turb), sigmay_d)
    
            ! vertical spread at discontinuity point
            call sigmaz_func(discontinuity_point, x0, kz_local(turb), rotorDiameter(turb), sigmaz_d)
            
            ! horizontal spread at far wake onset point
            call sigmay_func(x0, x0, ky_local(turb), rotorDiameter(turb), yaw(turb), sigmay_0)
    
            ! vertical spread at at far wake onset point
            call sigmaz_func(x0, x0, kz_local(turb), rotorDiameter(turb), sigmaz_0)
            
            ! calculate wake spread in horizontal at point of interest
            call sigma_spread_func(x, x0, ky_local(turb), sigmay_0, sigmay_d, 0.0_dp, 1.0_dp,  sigmay)
            
            ! calculate wake spread in vertical at point of interest
            call sigma_spread_func(x, x0, kz_local(turb), sigmaz_0, sigmaz_d, 0.0_dp, 1.0_dp, sigmaz)
            
            ! calculate new spread for WEC in y (horizontal)
            call sigma_spread_func(x, x0, ky_local(turb), sigmay_0, sigmay_d, wec_spreading_angle, wec_factor, sigmay_spread)
            
            ! calculate new spread for WEC in z (horizontal)
            call sigma_spread_func(x, x0, kz_local(turb), sigmaz_0, sigmaz_d, wec_spreading_angle, wec_factor, sigmaz_spread)

            ! calculate new spread for WEC in y (horizontal) at onset of far wake
            call sigma_spread_func(x0, x0, ky_local(turb), sigmay_0, sigmay_d, wec_spreading_angle, wec_factor, sigmay_0_spread)

            ! calculate new spread for WEC in z (horizontal) at onset of far wake
            call sigma_spread_func(x0, x0, kz_local(turb), sigmaz_0, sigmaz_d, wec_spreading_angle, wec_factor, sigmaz_0_spread)
            
            ! determine the initial wake angle at the onset of far wake
            call theta_c_0_func(yaw(turb), Ct_local(turb), theta_c_0)
            
            ! horizontal cross-wind wake displacement from hub
            call wake_offset_func(x, rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                 & ky_local(turb), kz_local(turb), &
                                 Ct_local(turb), sigmay, sigmaz, wake_offset)
            ! print *, x
            ! cross wind distance from point location to upstream turbine wake center
            deltay = pointY - (turbineYw(turb) + wake_offset)

            ! vertical distance from upstream hub height to height of point of interest
            deltaz = pointZ - turbineZ(turb)

            if (x > x0) then
                ! velocity difference in the wake
                call deltav_func(deltay, deltaz, Ct_local(turb), yaw(turb), &
                                & sigmay, sigmaz, rotorDiameter(turb), & 
                                & wake_model_version, kz_local(turb), x, &
                                & wec_factor, sigmay_spread, sigmaz_spread, deltav)
                                
            else
                ! velocity deficit in the nearwake (linear model)
                call deltav_near_wake_lin_func(deltay, deltaz, &
                                 & Ct_local(turb), yaw(turb), sigmay_0, sigmaz_0, x0, & 
                                 & rotorDiameter(turb), x, discontinuity_point, & 
                                 & sigmay_d, sigmaz_d, wake_model_version, &
                                 & kz_local(turb), x0, sigmay_spread, &
                                 & sigmaz_spread, sigmay_0_spread, &
                                 & sigmaz_0_spread, wec_factor, WECH, deltav)
                                 
            end if

            ! save deficit sum in holder for AD purposes
            old_deficit_sum = deficit_sum
            
            ! combine deficits according to selected wake combination method
            call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                       wake_combination_method, old_deficit_sum, deficit_sum)
        
        end if
    
    end do
    
    ! find velocity at point without shear
    point_velocity = wind_speed - deficit_sum
    
    ! adjust sample point velocity for shear
    call wind_shear_func(pointZ, point_velocity, z_ref, z_0, shear_exp, point_velocity_with_shear)
    
end subroutine point_velocity_with_shear_func

! calculates the onset of far-wake conditions
subroutine x0_func(rotor_diameter, yaw, Ct, alpha, TI, beta, x0)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: rotor_diameter, yaw, Ct, alpha, TI, beta

    ! out
    real(dp), intent(out) :: x0

    intrinsic cos, sqrt
                            

    ! determine the onset location of far wake
    x0 = rotor_diameter * (cos(yaw) * (1.0_dp + sqrt(1.0_dp - Ct)) / &
                                (sqrt(2.0_dp) * (alpha * TI + beta * &
                                                & (1.0_dp - sqrt(1.0_dp - Ct)))))
    
end subroutine x0_func


! calculates the wake angle at the onset of far wake conditions
subroutine theta_c_0_func(yaw, Ct, theta_c_0)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: yaw, Ct

    ! out
    real(dp), intent(out) :: theta_c_0

    intrinsic cos, sqrt
    
    ! determine the initial wake angle at the onset of far wake
    theta_c_0 = 0.3_dp * yaw * (1.0_dp - sqrt(1.0_dp - Ct * cos(yaw))) / cos(yaw)

end subroutine theta_c_0_func


! calculates the horizontal spread of the wake at a given distance from the onset of far 
! wake condition
subroutine sigmay_func(x, x0, ky, rotor_diameter, yaw, sigmay)
    
    implicit none

    ! define precision to be the standard for a double precision on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: ky, x, x0, rotor_diameter, yaw

    ! out
    real(dp), intent(out) :: sigmay

    intrinsic cos, sqrt
    
    ! horizontal spread
    sigmay = ky * (x-x0) + rotor_diameter * cos(yaw) / sqrt(8.0_dp)
    
end subroutine sigmay_func
    
    
! calculates the vertical spread of the wake at a given distance from the onset of far 
! wake condition
subroutine sigmaz_func(x, x0, kz, rotor_diameter, sigmaz)
    
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: kz, x, x0, rotor_diameter

    ! out
    real(dp), intent(out) :: sigmaz

    ! load necessary intrinsic functions
    intrinsic sqrt
    
    ! vertical spread
    sigmaz = kz * (x-x0) + rotor_diameter / sqrt(8.0_dp)
    
end subroutine sigmaz_func

subroutine sigma_spread_func(x, x0, k, sigma_0, sigma_d, wec_spreading_angle, wec_factor, sigma_spread)

    implicit none
    
    !! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: x, x0, k, sigma_0, sigma_d, wec_spreading_angle, wec_factor 

    ! out
    real(dp), intent(out) :: sigma_spread
    
    ! local
    real(dp) :: k_near_wake, sigma_0_new, k_spread
    real(dp), parameter :: pi = 3.141592653589793_dp
    
    intrinsic tan, atan
    
    ! check if spreading angle is too high
    if (wec_spreading_angle*pi/180.0_dp .ge. pi/2.0_dp) then
        print *, "WEC angle factor is too high, must be less than 90 deg "
        stop 1 
    end if
    
    ! get slope of wake expansion in the near wake
    k_near_wake = (sigma_0 - sigma_d) / x0
    k_spread = tan(wec_spreading_angle*pi/180.0_dp)
    if (k_spread .gt. k_near_wake) then
        k_near_wake = k_spread
    end if
    
    ! get new value for wake spread at the point of far wake onset
    sigma_0_new = k_near_wake * x0 + sigma_d

    ! get the wake spread at the point of interest
    if ((x .ge. x0) .and. (k .gt. k_near_wake)) then
        ! for points further downstream than the point of far wake onset and low spreading angles
        sigma_spread = wec_factor*(k * (x - x0) + sigma_0_new)
    else if (x .ge. 0.0_dp) then
        ! for points in the near wake and/or with high spreading angles
        sigma_spread = wec_factor*(k_near_wake * x + sigma_d)
    else
        ! for when the point is not in a wake
        sigma_spread = 0.0_dp
    end if
    
end subroutine sigma_spread_func

! calculates the horizontal distance from the wake center to the hub of the turbine making
! the wake
subroutine wake_offset_func(x, rotor_diameter, theta_c_0, x0, yaw, ky, kz, Ct, sigmay, &
                            & sigmaz, wake_offset)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: x, rotor_diameter, theta_c_0, x0, yaw, ky, kz, Ct, sigmay
    real(dp), intent(in) :: sigmaz
    
    ! local
    real(dp) :: a, b, c, d, e, f, g

    ! out
    real(dp), intent(out) :: wake_offset

    intrinsic cos, sqrt, log
    
    if (x < x0) then
    
        wake_offset = theta_c_0*x
    
    else
    
        a = theta_c_0*x0
        b = rotor_diameter*theta_c_0/14.7_dp
        c = sqrt(cos(yaw)/(ky*kz*Ct))
        d = 2.9_dp+1.3_dp*sqrt(1.0_dp - Ct)-Ct
        e = 1.6_dp*sqrt(8.0_dp*sigmay*sigmaz/((rotor_diameter**2)*cos(yaw)))
        f = (1.6_dp+sqrt(Ct))*(e-sqrt(Ct))
        g = (1.6_dp-sqrt(Ct))*(e+sqrt(Ct))
        
        wake_offset = a + b * c * d * log(f/g)

    end if
    
end subroutine wake_offset_func


! calculates the velocity difference between hub velocity and free stream for a given wake
! for use in the far wake region
subroutine deltav_func(deltay, deltaz, Ct, yaw, sigmay, sigmaz, & 
                      & rotor_diameter_ust, version, k, deltax, wec_factor, sigmay_spread, sigmaz_spread, deltav)
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter_ust, wec_factor, sigmay_spread, sigmaz_spread
    real(dp), intent(in) :: k, deltax    ! only for 2014 version
    integer, intent(in) :: version
    
    ! local
    real(dp) :: beta_2014, epsilon_2014 ! only for 2014 version
     
    ! out
    real(dp), intent(out) :: deltav

    ! load intrinsic functions
    intrinsic cos, sqrt, exp
    
    !print *, "rotor_diameter in deltav entry", rotor_diameter_ust
!     print *, 'wake model version in deltav: ', version

    if (version == 2014) then
        !print *, "in 2014 version"
        beta_2014 = 0.5_dp*(1.0_dp + sqrt(1.0_dp - Ct))/sqrt(1.0_dp - Ct)
        epsilon_2014 = 0.2_dp*sqrt(beta_2014)
        
       ! print *, "beta = ", beta_2014, "epsilon = ", epsilon_2014
       ! print *, "k, deltax: ", k, deltax
       ! print *, "term: ", Ct                                                   &
!                            / (8.0_dp * (k*deltax/rotor_diameter_ust+epsilon_2014)**2)
        deltav = (                                                                       &
            (1.0_dp - sqrt(1.0_dp - Ct                                                   &
                           / (8.0_dp * ((k*deltax/rotor_diameter_ust)+epsilon_2014)**2)))* &
            exp((-1.0_dp/(2.0_dp*((k*deltax/rotor_diameter_ust) + epsilon_2014)**2))*      & 
            ((deltaz/(wec_factor*rotor_diameter_ust))**2 + (deltay/(wec_factor*rotor_diameter_ust))**2))           &
        )
       ! print *, "deltav 2014 = ", deltav
    else if (version == 2016) then
        ! velocity difference in the wake at each sample point
        deltav = (                                                                    &
            (1.0_dp - sqrt(1.0_dp - Ct *                                                         &
                           cos(yaw) / (8.0_dp * sigmay * sigmaz / (rotor_diameter_ust ** 2)))) *     &
            exp(-0.5_dp * (deltay / (sigmay_spread)) ** 2) * exp(-0.5_dp * (deltaz / (sigmaz_spread)) ** 2)&
        )
    else
        print *, "Invalid Bastankhah and Porte Agel model version. Must be 2014 or 2016. ", version, " was given."
        stop 1
    end if 
    
    !print *, "rotor_diameter in deltav exit", rotor_diameter_ust

end subroutine deltav_func


! calculates the velocity difference between hub velocity and free stream for a given wake
! for use in the near wake region only
subroutine deltav_near_wake_lin_func(deltay, deltaz, Ct, yaw, &
                                 & sigmay_0, sigmaz_0, x0, rotor_diameter_ust, x, &
                                 & discontinuity_point, sigmay_d, sigmaz_d, version, k_2014, &
                                 & deltaxd_2014, sigmay_spread, sigmaz_spread, sigmay_0_spread, sigmaz_0_spread, &
                                 & wec_factor_2014, WECH, deltav)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, Ct, yaw, sigmay_0, sigmaz_0, x0, sigmay_0_spread, sigmaz_0_spread
    real(dp), intent(in) :: rotor_diameter_ust, sigmay_spread, sigmaz_spread
    real(dp), intent(in) :: x, discontinuity_point, sigmay_d, sigmaz_d
    real(dp), intent(in) :: k_2014, deltaxd_2014, wec_factor_2014    ! only for 2014 version
    integer, intent(in) :: version, WECH
    
    ! local
    real(dp) :: deltav0m, deltavdm, sigmay, sigmaz
    real(dp) :: beta_2014, epsilon_2014 ! only for 2014 version

    ! out
    real(dp), intent(out) :: deltav

    ! load intrinsic functions
    intrinsic cos, sqrt, exp

    if (version == 2014) then !TODO fix 2014 version
        if (yaw > 0.0_dp) then
            print *, "model version 2014 may only be used when yaw=0"
            stop 1
        end if
        beta_2014 = 0.5_dp*(1.0_dp + sqrt(1.0_dp - Ct))/sqrt(1.0_dp - Ct)
        epsilon_2014 = 0.2_dp*sqrt(beta_2014)
        
        ! magnitude term of gaussian at x0
        deltav0m = (1.0_dp - sqrt(1.0_dp - Ct                                            &
                           / (8.0_dp * (k_2014*x0/rotor_diameter_ust+epsilon_2014)**2)))
        
        ! initialize the gaussian magnitude term at the rotor for the linear interpolation
        deltavdm = (1.0_dp - sqrt(1.0_dp - Ct                                            &
                           / (8.0_dp * (k_2014*discontinuity_point/rotor_diameter_ust+epsilon_2014)**2)))
        
        ! linearized gaussian magnitude term for near wake
        deltav = (                                                                       &
             (((deltav0m - deltavdm)/x0) * x + deltavdm) *                &
            exp((-1.0_dp/(2.0_dp*(k_2014*x/rotor_diameter_ust + epsilon_2014)**2))*      & 
            ((deltaz/(wec_factor_2014*rotor_diameter_ust))**2 + (deltay/(wec_factor_2014*rotor_diameter_ust))**2))           &
        )
        
    else if (version == 2016) then

        ! magnitude term of gaussian at x0
        deltav0m = ((1.0_dp - sqrt(1.0_dp - Ct *                          &
                    cos(yaw) / (8.0_dp * sigmay_0 * sigmaz_0/(rotor_diameter_ust ** 2)))))
        ! initialize the gaussian magnitude term at the rotor for the linear interpolation
        deltavdm = ((1.0_dp - sqrt(1.0_dp - Ct *                          &
                    cos(yaw) / (8.0_dp * sigmay_d * sigmaz_d/(rotor_diameter_ust ** 2)))))

        if (WECH == 0) then
            ! linearized gaussian magnitude term for near wake
            deltav = (((deltav0m - deltavdm)/x0) * x + deltavdm) *       &
                exp(-0.5_dp * (deltay / (sigmay_spread)) ** 2) * &
                exp(-0.5_dp * (deltaz / (sigmaz_spread)) ** 2)
        else
            ! linearized gaussian magnitude term for near wake for WECH
            sigmay = ((sigmay_0_spread - sigmay_0)/(x0)) * x + sigmay_0
            sigmaz = ((sigmaz_0_spread - sigmay_0)/(x0)) * x + sigmaz_0
            deltav = (((deltav0m - deltavdm)/x0) * x + deltavdm) *       &
                exp(-0.5_dp * (deltay / (sigmay)) ** 2) * &
                exp(-0.5_dp * (deltaz / (sigmaz)) ** 2)
        end if
            
    else
    
        print *, "Invalid Bastankhah and Porte Agel model version. Must be 2014 or 2016. ", version, " was given."
        stop 1
    end if
                
end subroutine deltav_near_wake_lin_func

! calculates the overlap area between a given wake and a rotor area
subroutine overlap_area_func(turbine_y, turbine_z, rotor_diameter, &
                            wake_center_y, wake_center_z, wake_diameter, &
                            wake_overlap)

    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: turbine_y, turbine_z, rotor_diameter
    real(dp), intent(in) :: wake_center_y, wake_center_z, wake_diameter
    
    ! out    
    real(dp), intent(out) :: wake_overlap
    
    ! local
    real(dp), parameter :: pi = 3.141592653589793_dp, tol = 0.000001_dp
    real(dp) :: OVdYd, OVr, OVRR, OVL, OVz, OVz2
    
    ! load intrinsic functions
    intrinsic acos, sqrt
    
!     print *, turbine_y, turbine_z, rotor_diameter, &
!                             wake_center_y, wake_center_z, wake_diameter, &
!                             wake_overlap
    
   ! distance between wake center and rotor center
    if ((wake_center_z > (turbine_z + tol)) .or. (wake_center_z < (turbine_z - tol))) then
        OVdYd = sqrt((wake_center_y-turbine_y)**2_dp + (wake_center_z - turbine_z)**2_dp)
    else if (wake_center_y > (turbine_y + tol)) then! potential source of gradient issues, abs() did not cause a problem in FLORIS
        OVdYd = wake_center_y - turbine_y
    else if (turbine_y > (wake_center_y + tol)) then
        OVdYd = turbine_y - wake_center_y
    else
        OVdYd = 0.0_dp
    end if
    
    !print *, "OVdYd: ", OVdYd
    ! find rotor radius
    OVr = rotor_diameter/2.0_dp
    !print *, "OVr: ", OVr
    
    ! find wake radius
    OVRR = wake_diameter/2.0_dp
    !print *, "OVRR: ", OVRR
    
    ! make sure the distance from wake center to turbine hub is positive
    ! OVdYd = abs(OVdYd) !!! commented out since change to 2D distance (y,z) will always be positive
    
    ! calculate the distance from the wake center to the line perpendicular to the 
    ! line between the two circle intersection points
    !if (OVdYd >= 0.0_dp + tol) then ! check case to avoid division by zero
!     print *, "OVdYd ", OVdYd
    ! if (OVdYd >= 0.0_dp + tol) then ! check case to avoid division by zero
!         OVL = (-OVr*OVr+OVRR*OVRR+OVdYd*OVdYd)/(2.0_dp*OVdYd)
! !         print *, "OVdYd, OVL: ", OVdYd, OVL
!     else
!         OVL = 0.0_dp
!     end if
! 
!     OVz = OVRR*OVRR-OVL*OVL
! 
!     ! Finish calculating the distance from the intersection line to the outer edge of the wake
!     !if (OVz > 0.0_dp + tol) then
!     if (OVz > 0.0_dp + tol) then
!         OVz = sqrt(OVz)
!     else
!         OVz = 0.0_dp
!     end if
    
    !print *, "OVRR, OVL, OVRR, OVr, OVdYd, OVz ", OVRR, OVL, OVRR, OVr, OVdYd, OVz
    
    

    ! if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake
!         !print *, "OVL: ", OVL
!         if (OVL < OVRR .and. (OVdYd-OVL) < OVr) then
! !         if (OVdYd > 0.0_dp + tol) then
! !         if ((OVdYd > 0.0_dp) .and. (OVdYd > (OVRR - OVr))) then
!             ! print *, "acos(OVL/OVRR), acos((OVdYd-OVL)/OVr), OVRR, OVL, OVr, OVdYd, OVL/OVRR, (OVdYd-OVL)/OVr ", &
! !     & acos(OVL/OVRR), acos((OVdYd-OVL)/OVr), OVRR, OVL, OVr, OVdYd, OVL/OVRR, (OVdYd-OVL)/OVr
!             wake_overlap = OVRR*OVRR*acos(OVL/OVRR) + OVr*OVr*acos((OVdYd-OVL)/OVr) - OVdYd*OVz
!         else if (OVRR > OVr) then
!             wake_overlap = pi*OVr*OVr
!             !print *, "wake ovl: ", wake_overlap
!         else
!             wake_overlap = pi*OVRR*OVRR
!         end if
!     else
!         wake_overlap = 0.0_dp
!     end if

    ! determine if there is overlap
    if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake zone

        ! check that turbine and wake centers are not perfectly aligned
        if (OVdYd > (0.0_dp + tol)) then
        
            ! check if the rotor is wholly contained in the wake
            if ((OVdYd + OVr) < OVRR + tol) then 
                wake_overlap = pi*OVr*OVr
!                 print *, "1"
            ! check if the wake is wholly contained in the rotor swept area
            else if ((OVdYd + OVRR) < OVr + tol) then
                wake_overlap = pi*OVRR*OVRR
!                 print *, "2"
            else
            
                ! calculate the distance from the wake center to the chord connecting the lens
                ! cusps
                OVL = (-OVr*OVr+OVRR*OVRR+OVdYd*OVdYd)/(2.0_dp*OVdYd)

                OVz = sqrt(OVRR*OVRR-OVL*OVL)
                OVz2 = sqrt(OVr*OVr-(OVdYd-OVL)*(OVdYd-OVL))
            
                wake_overlap = OVRR*OVRR*acos(OVL/OVRR) + OVr*OVr*acos((OVdYd-OVL)/OVr) - &
                               & OVL*OVz - (OVdYd-OVL)*OVz2
!                 print *, OVRR, OVr, OVdYd, OVL, OVz, OVz2
!                 print *, "3"
            end if
        
        ! perfect overlap case where the wake is larger than the rotor
        else if (OVRR > OVr) then
            wake_overlap = pi*OVr*OVr
!             print *, "4"
            
        ! perfect overlap case where the rotor is larger than the wake
        else
            wake_overlap = pi*OVRR*OVRR
!             print *, "5"
        end if
        
    ! case with no overlap
    else
        wake_overlap = 0.0_dp
    end if
    
!     print *, "wake overlap in func: ", wake_overlap/(pi*OVr**2)
!     print *, "wake overlap in func: ", wake_overlap/(pi*OVRR**2)
    
    if ((wake_overlap/(pi*OVr*OVr) > (1.0_dp + tol)) .or. (wake_overlap/(pi*OVRR*OVRR) > (1.0_dp + tol))) then
        print *, "wake overlap in func: ", wake_overlap/(pi*OVr*OVr)
        print *, "wake overlap in func: ", wake_overlap/(pi*OVRR*OVRR)
        STOP 1
    end if
                             
end subroutine overlap_area_func

 ! combines wakes using various methods
subroutine wake_combination_func(wind_speed, turb_inflow, deltav,                  &
                                 wake_combination_method, old_deficit_sum, new_deficit_sum)
                                 
    ! combines wakes to calculate velocity at a given turbine
    ! wind_speed                = Free stream velocity
    ! turb_inflow               = Effective velocity as seen by the upstream rotor
    ! deltav                    = Velocity deficit percentage for current turbine pair
    ! wake_combination_method   = Use for selecting which method to use for wake combo
    ! deficit_sum (in)          = Combined deficits prior to including the current deltav
    ! deficit_sum (out)         = Combined deficits after to including the current deltav
    
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: wind_speed, turb_inflow, deltav, old_deficit_sum
    integer, intent(in) :: wake_combination_method
    
    ! out    
    real(dp), intent(out) :: new_deficit_sum
    
    ! intrinsic functions
    intrinsic sqrt
    
    ! freestream linear superposition (Lissaman 1979)
    if (wake_combination_method == 0) then
        new_deficit_sum = old_deficit_sum + wind_speed*deltav

    ! local velocity linear superposition (Niayifar and Porte Agel 2015, 2016)
    else if (wake_combination_method == 1) then
        new_deficit_sum = old_deficit_sum + turb_inflow*deltav
        !print *, "here"
        
    ! sum of squares freestream superposition (Katic et al. 1986)
    else if (wake_combination_method == 2) then 
        new_deficit_sum = sqrt(old_deficit_sum**2 + (wind_speed*deltav)**2)
    
    ! sum of squares local velocity superposition (Voutsinas 1990)
    else if (wake_combination_method == 3) then
        new_deficit_sum = sqrt(old_deficit_sum**2 + (turb_inflow*deltav)**2)
    
    ! wake combination method error
    else
        print *, "Invalid wake combination method. Must be one of [0,1,2,3]."
        stop 1
    end if                       
    
end subroutine wake_combination_func

! combines wakes using various methods
subroutine added_ti_func(TI, Ct_ust, x, k_star_ust, rotor_diameter_ust, rotor_diameter_dst, & 
                        & deltay, wake_height, turbine_height, sm_smoothing, TI_ust, &
                        & TI_calculation_method, TI_area_ratio_in, TI_dst_in, TI_area_ratio, TI_dst)
                                 
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: Ct_ust, x, k_star_ust, rotor_diameter_ust, rotor_diameter_dst
    real(dp), intent(in) :: deltay, wake_height, turbine_height, sm_smoothing
    real(dp), intent(in) :: TI_ust, TI, TI_area_ratio_in, TI_dst_in
    integer, intent(in) :: TI_calculation_method
    
    ! local
    real(dp) :: axial_induction_ust, beta, epsilon, sigma, wake_diameter, wake_overlap
    real(dp) :: TI_added, TI_tmp, rotor_area_dst, TI_area_ratio_tmp
    real(dp), parameter :: pi = 3.141592653589793_dp
    
    ! out  
    real(dp), intent(out) :: TI_dst, TI_area_ratio
    
    ! intrinsic functions
    intrinsic sqrt
    
    ! initialize output variables
    TI_area_ratio = TI_area_ratio_in
    TI_dst = TI_dst_in
    
    ! initialize wake overlap to zero
    wake_overlap = 0.0_dp
    
    !print *, "TI_dst in: ", TI_dst
    ! Niayifar and Porte Agel 2015, 2016 (adjusted by Annoni and Thomas for SOWFA match 
    ! and optimization)
    if (TI_calculation_method == 1) then
    
        ! calculate axial induction based on the Ct value
        call ct_to_axial_ind_func(Ct_ust, axial_induction_ust)
        
        ! calculate BPA spread parameters Bastankhah and Porte Agel 2014
        beta = 0.5_dp*((1.0_dp + sqrt(1.0_dp - Ct_ust))/sqrt(1.0_dp - Ct_ust))
        epsilon = 0.2_dp*sqrt(beta)
        !print *, "epsilon = ", epsilon
        ! calculate wake spread for TI calcs
        sigma = k_star_ust*x + rotor_diameter_ust*epsilon
        wake_diameter = 4.0_dp*sigma
        !print *, "sigma = ", sigma
        ! calculate wake overlap ratio
        call overlap_area_func(deltay, turbine_height, rotor_diameter_dst, &
                            0.0_dp, wake_height, wake_diameter, &
                            wake_overlap)
        !print *, "wake_overlap = ", wake_overlap   
        ! Calculate the turbulence added to the inflow of the downstream turbine by the 
        ! wake of the upstream turbine
        TI_added = 0.73_dp*(axial_induction_ust**0.8325_dp)*(TI_ust**0.0325_dp)* & 
                    ((x/rotor_diameter_ust)**(-0.32_dp))
        !print *, "TI_added = ", TI_added
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2_dp
        ! Calculate the total turbulence intensity at the downstream turbine
        !sum_of_squares = TI_dst**2 + (TI_added*wake_overlap)**2
        ! print *, "sum of squares = ", sum_of_squares
!         TI_dst = sqrt(sum_of_squares)
!         !print *, "TI_dst = ", TI_dst
        TI_dst = sqrt(TI_dst_in**2.0_dp + (TI_added*wake_overlap/rotor_area_dst)**2.0_dp)
        
    
    ! Niayifar and Porte Agel 2015, 2016
    else if (TI_calculation_method == 2) then
    
        ! calculate axial induction based on the Ct value
        call ct_to_axial_ind_func(Ct_ust, axial_induction_ust)
        
        ! calculate BPA spread parameters Bastankhah and Porte Agel 2014
        beta = 0.5_dp*((1.0_dp + sqrt(1.0_dp - Ct_ust))/sqrt(1.0_dp - Ct_ust))
        epsilon = 0.2_dp*sqrt(beta)
        
        ! calculate wake spread for TI calcs
        sigma = k_star_ust*x + rotor_diameter_ust*epsilon
        wake_diameter = 4.0_dp*sigma
        
        ! calculate wake overlap ratio
        call overlap_area_func(deltay, turbine_height, rotor_diameter_dst, &
                            0.0_dp, wake_height, wake_diameter, &
                            wake_overlap)
                            
        ! Calculate the turbulence added to the inflow of the downstream turbine by the 
        ! wake of the upstream turbine
        TI_added = 0.73_dp*(axial_induction_ust**0.8325_dp)*(TI_ust**0.0325_dp)* & 
                    ((x/rotor_diameter_ust)**(-0.32_dp))
        
        ! Calculate the total turbulence intensity at the downstream turbine based on 
        ! current upstream turbine
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2_dp
        TI_tmp = sqrt(TI**2.0_dp + (TI_added*(wake_overlap/rotor_area_dst))**2.0_dp)
        
        ! Check if this is the max and use it if it is
        if (TI_tmp > TI_dst_in) then
!            print *, "TI_tmp > TI_dst"
           TI_dst = TI_tmp
        end if
        
    ! Niayifar and Porte Agel 2015, 2016 with smooth max
     else if (TI_calculation_method == 3) then
     
        ! calculate axial induction based on the Ct value
        call ct_to_axial_ind_func(Ct_ust, axial_induction_ust)
        
        ! calculate BPA spread parameters Bastankhah and Porte Agel 2014
        beta = 0.5_dp*((1.0_dp + sqrt(1.0_dp - Ct_ust))/sqrt(1.0_dp - Ct_ust))
        epsilon = 0.2_dp*sqrt(beta)
        
        ! calculate wake spread for TI calcs
        sigma = k_star_ust*x + rotor_diameter_ust*epsilon
        wake_diameter = 4.0_dp*sigma
        
!         print *, "sigma, k_star_ust, x, rotor_diameter_ust, epsilon ", sigma, k_star_ust, x, rotor_diameter_ust, epsilon
        
        ! print *, "deltay, turbine_height, rotor_diameter_dst, wake_height, wake_diameter", &
!                 & deltay, turbine_height, rotor_diameter_dst, &
!                             wake_height, wake_diameter
        
        ! calculate wake overlap ratio
        call overlap_area_func(deltay, turbine_height, rotor_diameter_dst, &
                            0.0_dp, wake_height, wake_diameter, &
                            wake_overlap)
                            
        ! Calculate the turbulence added to the inflow of the downstream turbine by the 
        ! wake of the upstream turbine
        TI_added = 0.73_dp*(axial_induction_ust**0.8325_dp)*(TI_ust**0.0325_dp)* & 
                    ((x/rotor_diameter_ust)**(-0.32_dp))
        
        ! Calculate the total turbulence intensity at the downstream turbine based on 
        ! current upstream turbine
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2_dp
        TI_tmp = sqrt(TI**2.0_dp + (TI_added*(wake_overlap/rotor_area_dst))**2.0_dp)
        
        !print *, "TI, TI_added, wake_overlap, rotor_area_dst: ", TI, TI_added, wake_overlap, rotor_area_dst
        
        ! Check if this is the max and use it if it is
        !if (TI_tmp > TI_dst) then
        !    TI_dst = TI_tmp
        !end if
!         print *, "before: ", TI_dst, TI_tmp
!         TI_dst_in = TI_dst
        call smooth_max(sm_smoothing, TI_dst_in, TI_tmp, TI_dst)
!         print *, "after:: ", TI_dst, TI_tmp

    ! Niayifar and Porte Agel 2015, 2016 using max on area TI ratio
    else if (TI_calculation_method == 4) then
    
        ! calculate axial induction based on the Ct value
        call ct_to_axial_ind_func(Ct_ust, axial_induction_ust)
        
        ! calculate BPA spread parameters Bastankhah and Porte Agel 2014
        beta = 0.5_dp*((1.0_dp + sqrt(1.0_dp - Ct_ust))/sqrt(1.0_dp - Ct_ust))
        epsilon = 0.2_dp*sqrt(beta)
        
        ! calculate wake spread for TI calcs
        sigma = k_star_ust*x + rotor_diameter_ust*epsilon
        wake_diameter = 4.0_dp*sigma
        
        ! calculate wake overlap ratio
        call overlap_area_func(deltay, turbine_height, rotor_diameter_dst, &
                            0.0_dp, wake_height, wake_diameter, &
                            wake_overlap)
                            
        ! Calculate the turbulence added to the inflow of the downstream turbine by the 
        ! wake of the upstream turbine
        TI_added = 0.73_dp*(axial_induction_ust**0.8325_dp)*(TI_ust**0.0325_dp)* & 
                    ((x/rotor_diameter_ust)**(-0.32_dp))
        
        ! Calculate the total turbulence intensity at the downstream turbine based on 
        ! current upstream turbine
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2_dp
        TI_area_ratio_tmp = TI_added*(wake_overlap/rotor_area_dst)
        
        ! Check if this is the max and use it if it is
        if (TI_area_ratio_tmp > TI_area_ratio_in) then
!            print *, "ti_area_ratio_tmp > ti_area_ratio"
           !TI_dst = TI_tmp
           TI_area_ratio = TI_area_ratio_tmp
           TI_dst = sqrt(TI**2.0_dp + (TI_area_ratio)**2.0_dp)
           
        end if
    
    ! Niayifar and Porte Agel 2015, 2016 using smooth max on area TI ratio
    else if (TI_calculation_method == 5) then
    
        ! calculate axial induction based on the Ct value
        call ct_to_axial_ind_func(Ct_ust, axial_induction_ust)
        
        ! calculate BPA spread parameters Bastankhah and Porte Agel 2014
        beta = 0.5_dp*((1.0_dp + sqrt(1.0_dp - Ct_ust))/sqrt(1.0_dp - Ct_ust))
        epsilon = 0.2_dp*sqrt(beta)
        
        ! calculate wake spread for TI calcs
        sigma = k_star_ust*x + rotor_diameter_ust*epsilon
        wake_diameter = 4.0_dp*sigma
        
        ! calculate wake overlap ratio
        call overlap_area_func(deltay, turbine_height, rotor_diameter_dst, &
                            0.0_dp, wake_height, wake_diameter, &
                             wake_overlap)
        ! only include turbines with area overlap in the softmax
        if (wake_overlap > 0.0_dp) then
        
            ! Calculate the turbulence added to the inflow of the downstream turbine by the 
            ! wake of the upstream turbine
            TI_added = 0.73_dp*(axial_induction_ust**0.8325_dp)*(TI_ust**0.0325_dp)* & 
                        ((x/rotor_diameter_ust)**(-0.32_dp))
        

            rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2_dp
            TI_area_ratio_tmp = TI_added*(wake_overlap/rotor_area_dst)
            !TI_tmp = sqrt(TI**2.0_dp + (TI_added*(wake_overlap/rotor_area_dst))**2.0_dp)
        
            ! Run through the smooth max to get an approximation of the true max TI area ratio
            call smooth_max(sm_smoothing, TI_area_ratio_in, TI_area_ratio_tmp, TI_area_ratio)
            
            ! Calculate the total turbulence intensity at the downstream turbine based on 
            ! the result of the smooth max function
            TI_dst = sqrt(TI**2.0_dp + TI_area_ratio**2.0_dp)
            
        end if
    
    ! wake combination method error 
    else
        print *, "Invalid added TI calculation method. Must be one of [0,1,2,3,4,5]."
        stop 1
    end if            
    
    !print *, "ratio: ", wake_overlap/rotor_area_dst
    !print *, "Dr, Dw: ", rotor_diameter_dst, wake_diameter
    !print *, "Ar, Aol: ", rotor_area_dst, wake_overlap          
    
end subroutine added_ti_func

! compute wake spread parameter based on local turbulence intensity
subroutine k_star_func(TI_ust, k_star_ust)
                                 
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: TI_ust
    
    ! out  
    real(dp), intent(out) :: k_star_ust
    
    ! calculate wake spread parameter from Niayifar and Porte Agel (2015, 2016)
    k_star_ust = 0.3837*TI_ust+0.003678
    
end subroutine k_star_func

! calculate axial induction from Ct
subroutine ct_to_axial_ind_func(CT, axial_induction)
    
    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: CT

    ! out
    real(dp), intent(out) :: axial_induction

    ! initialize axial induction to zero
    axial_induction = 0.0_dp

    ! calculate axial induction
    if (CT > 0.96) then  ! Glauert condition
        axial_induction = 0.143_dp + sqrt(0.0203_dp-0.6427_dp*(0.889_dp - CT))
    else
        axial_induction = 0.5_dp*(1.0_dp-sqrt(1.0_dp-CT))
    end if
    
end subroutine ct_to_axial_ind_func

! adjust wind speed for wind shear
subroutine wind_shear_func(point_z, u_ref, z_ref, z_0, shear_exp, adjusted_wind_speed)

    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: point_z, u_ref, z_ref, z_0, shear_exp

    ! out
    real(dp), intent(out) :: adjusted_wind_speed

    ! initialize adjusted wind speed to zero
    adjusted_wind_speed = 0.0_dp

    ! check that the point of interest is above ground level
    if (point_z >= z_0) then
        ! adjusted wind speed for wind shear if point is above ground
        adjusted_wind_speed = u_ref*((point_z-z_0)/(z_ref-z_0))**shear_exp
    else 
        ! if the point of interest is below ground, set the wind speed to 0.0
        adjusted_wind_speed = 0.0_dp
    end if
    
end subroutine wind_shear_func


! calculate the point where the Bastankhah and Porte Agel wake model becomes undefined
subroutine discontinuity_point_func(x0, rotor_diameter, ky, kz, yaw, Ct, discontinuity_point)
    
    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: x0, rotor_diameter, ky, kz, yaw, Ct
    
    ! local
    real(dp) :: a, b, c

    ! out
    real(dp), intent(out) :: discontinuity_point
    
    intrinsic cos, sqrt
    
    ! for clarity, break out the terms in the equation
    a = ky + kz*cos(yaw)
    b = 4.0_dp * ky * kz * cos(yaw)*(Ct - 1.0_dp)
    c = 2.0_dp * sqrt(8.0_dp) * ky * kz

    ! distance from rotor to the last point where the wake model is undefined
    discontinuity_point = x0 + rotor_diameter * (a - sqrt(a**2 - b))/c
    
end subroutine discontinuity_point_func

subroutine smooth_max(s, x, y, g)

    ! based on John D. Cook's writings at 
    ! (1) https://www.johndcook.com/blog/2010/01/13/soft-maximum/
    ! and
    ! (2) https://www.johndcook.com/blog/2010/01/20/how-to-compute-the-soft-maximum/
    
    ! s controls the level of smoothing used in the smooth max
    ! x and y are the values to be compared
    
    ! g is the result

    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: s, x, y
    
    ! local
    real(dp) :: max_val, min_val
    
    ! out
    real(dp), intent(out) :: g
    
    intrinsic log, exp, max, min

    ! LogSumExponential Method - used this in the past
    ! g = (x*exp(s*x)+y*exp(s*y))/(exp(s*x)+exp(s*y))

    ! non-overflowing version of Smooth Max function (see ref 2 above)
    max_val = max(x, y)
    min_val = min(x, y)
    g = (log(1.0_dp + exp(s*(min_val - max_val)))+s*max_val)/s
    
end subroutine smooth_max

subroutine interpolation(nPoints, interp_type, x, y, xval, yval, dy0in, dy1in, usedyin)

    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nPoints, interp_type
    real(dp), dimension(nPoints), intent(in) :: x, y
    real(dp), intent(in) :: xval
    real(dp), intent(in):: dy0in, dy1in
    logical :: usedyin
    
    ! local
    integer :: idx
    real(dp) :: x0, x1, y0, dy0, y1, dy1
    
    ! out
    real(dp), intent(out) :: yval
    
!     print *, "in interpolation"
    
    ! if ((xval < x(1)) .or. (xval > x(nPoints))) then
!         print *, "interpolation point is out of bounds"
! !         STOP 1
!     end if

    if (usedyin .and. (interp_type == 1)) then
        print *, "end point derivatives may not be specified for linear interpolation"
        STOP 1
    end if
    
    if (xval < x(1)) then
        yval = y(1)
    else if (xval > x(nPoints)) then
        yval = y(nPoints)
    
    else
        idx = 1
    
        do while ((xval > x(idx)) .and. (idx <= nPoints))
            idx = idx + 1
        end do
    
        idx = idx - 1
        
        x0 = x(idx)
        x1 = x((idx + 1))
        y0 = y(idx)
        y1 = y((idx + 1))
    
        ! Hermite cubic piecewise spline interpolation
        if (interp_type == 0) then
    
            ! approximate derivative at left end of interval
            if (idx == 1) then
            
                if (usedyin) then
                    dy0 = dy0in
                else
                    dy0 = 0.0_dp
                endif
    
            else
                dy0 = (y(idx+1) - y(idx-1))/(x(idx+1) - x(idx-1))
            end if
    
            ! approximate derivative at the right end of interval
            if (idx >= nPoints-1) then
            
                if(usedyin)then
                    dy1 = dy1in
                else
                    dy1 = 0.0_dp
                endif
            else
                dy1 = (y(idx+2) - y(idx))/(x(idx+2) - x(idx))
            end if
    
            ! call Hermite spline routine
            call Hermite_Spline(xval, x0, x1, y0, dy0, y1, dy1, yval)
    
        ! linear interpolation
        else if (interp_type == 1) then
        
            yval = (xval-x0)*(y1-y0)/(x1-x0) + y0
        
        end if
    end if
    
!     print *, "yval = ", yval
    
end subroutine interpolation
    
    
subroutine Hermite_Spline(x, x0, x1, y0, dy0, y1, dy1, y)
    !    This function produces the y and dy values for a hermite cubic spline
    !    interpolating between two end points with known slopes
    !
    !    :param x: x position of output y
    !    :param x0: x position of upwind endpoint of spline
    !    :param x1: x position of downwind endpoint of spline
    !    :param y0: y position of upwind endpoint of spline
    !    :param dy0: slope at upwind endpoint of spline
    !    :param y1: y position of downwind endpoint of spline
    !    :param dy1: slope at downwind endpoint of spline
    !
    !    :return: y: y value of spline at location x
    
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: x, x0, x1, y0, dy0, y1, dy1
    
    ! out
    real(dp), intent(out) :: y !, dy_dx
    
    ! local
    real(dp) :: c3, c2, c1, c0

    ! initialize coefficients for parametric cubic spline
    c3 = (2.0_dp*(y1))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) - &
         (2.0_dp*(y0))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) + &
         (dy0)/(x0**2 - 2.0_dp*x0*x1 + x1**2) + &
         (dy1)/(x0**2 - 2.0_dp*x0*x1 + x1**2)
         
    c2 = (3.0_dp*(y0)*(x0 + x1))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) - &
         ((dy1)*(2.0_dp*x0 + x1))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - ((dy0)*(x0 + &
         2.0_dp*x1))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - (3.0_dp*(y1)*(x0 + x1))/(x0**3 - &
         3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3)
         
    c1 = ((dy0)*(x1**2 + 2.0_dp*x0*x1))/(x0**2 - 2.0_dp*x0*x1 + x1**2) + ((dy1)*(x0**2 + &
         2.0_dp*x1*x0))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - (6.0_dp*x0*x1*(y0))/(x0**3 - &
         3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3) + (6.0_dp*x0*x1*(y1))/(x0**3 - &
         3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - x1**3)
         
    c0 = ((y0)*(- x1**3 + 3.0_dp*x0*x1**2))/(x0**3 - 3.0_dp*x0**2*x1 + 3.0_dp*x0*x1**2 - &
         x1**3) - ((y1)*(- x0**3 + 3.0_dp*x1*x0**2))/(x0**3 - 3.0_dp*x0**2*x1 + &
         3.0_dp*x0*x1**2 - x1**3) - (x0*x1**2*(dy0))/(x0**2 - 2.0_dp*x0*x1 + x1**2) - &
         (x0**2*x1*(dy1))/(x0**2 - 2.0_dp*x0*x1 + x1**2)
!    print *, 'c3 = ', c3
!    print *, 'c2 = ', c2
!    print *, 'c1 = ', c1
!    print *, 'c0 = ', c0
    ! Solve for y and dy values at the given point
    y = c3*x**3 + c2*x**2 + c1*x + c0
    !dy_dx = c3*3*x**2 + c2*2*x + c1

end subroutine Hermite_Spline


 !    yd, n = _checkIfFloat(yd)
! 
!     y1 = (1-pct_offset)*ymax
!     y2 = (1+pct_offset)*ymax
! 
!     dy1 = (1-pct_offset)
!     dy2 = (1+pct_offset)
! 
!     if (maxmin == 1) then
!         f1 = y1
!         f2 = ymax
!         g1 = 1.0_dp
!         g2 = 0.0_dp
!         if (yd .ge. y2) then
!             idx_constant = False
!         else
!             idx_constant = True
!         end if
! 
!         df1 = dy1
!         df2 = 1.0_dp
! 
! 
!     else if (maxmin == 0) then
!         f1 = ymax
!         f2 = y2
!         g1 = 0.0_dp
!         g2 = 1.0_dp
!         if (yd .ge. y1) then
!             idx_constant = False
!         else
!             idx_constant = True
!         end if
! 
!         df1 = 1.0_dp
!         df2 = dy2
!         
!     end if
! 
!     f = CubicSplineSegment(y1, y2, f1, f2, g1, g2)
! 
!     # main region
!     ya = np.copy(yd)
!     if dyd is None:
!         dya_dyd = np.ones_like(yd)
!     else:
!         dya_dyd = np.copy(dyd)
! 
!     dya_dymax = np.zeros_like(ya)
! 
!     # cubic spline region
!     idx = np.logical_and(yd > y1, yd < y2)
!     ya[idx] = f.eval(yd[idx])
!     dya_dyd[idx] = f.eval_deriv(yd[idx])
!     dya_dymax[idx] = f.eval_deriv_params(yd[idx], dy1, dy2, df1, df2, 0.0, 0.0)
! 
!     # constant region
!     ya[idx_constant] = ymax
!     dya_dyd[idx_constant] = 0.0
!     dya_dymax[idx_constant] = 1.0
! 
!     if n == 1:
!         ya = ya[0]
!         dya_dyd = dya_dyd[0]
!         dya_dymax = dya_dymax[0]
! 
! 
!     return ya, dya_dyd, dya_dymax
! 
! 
! def smooth_max(yd, ymax, pct_offset=0.01, dyd=None):
!     """array max, uses cubic spline to smoothly transition.  derivatives with respect to array and max value.
!     width of transition can be controlled, and chain rules for differentiation"""
!     return _smooth_maxmin(yd, ymax, 'max', pct_offset, dyd)
! 
! 
! def smooth_min(yd, ymin, pct_offset=0.01, dyd=None):
!     """array min, uses cubic spline to smoothly transition.  derivatives with respect to array and min value.
!     width of transition can be controlled, and chain rules for differentiation"""
!     return _smooth_maxmin(yd, ymin, 'min', pct_offset, dyd)
! 
