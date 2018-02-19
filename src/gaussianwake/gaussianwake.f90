! Implementation of the Bastankhah and Porte Agel gaussian-shaped wind turbine wake 
! model (2016) with various farm modeling (TI and wake combination) methods included
! Created by Jared J. Thomas, 2017.
! FLight Optimization and Wind Laboratory (FLOW Lab)
! Brigham Young University

! implementation of the Bastankhah and Porte Agel (BPA) wake model for analysis
subroutine porteagel_analyze(nTurbines, nRotorPoints, nCtPoints, turbineXw, &
                             sorted_x_idx, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, TI, RotorPointsY, RotorPointsZ, &
                             z_ref, z_0, shear_exp, wake_combination_method, &
                             TI_calculation_method, calc_k_star, opt_exp_fac, print_ti, &
                             wake_model_version, interp_type, &
                             use_ct_curve, ct_curve_wind_speed, ct_curve_ct, wtVelocity)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter Ct yawDeg

    ! dependent variables: wtVelocity


    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nRotorPoints, nCtPoints
    integer, intent(in) :: wake_combination_method, TI_calculation_method, & 
                        &  wake_model_version, interp_type
    logical, intent(in) :: calc_k_star, print_ti, use_ct_curve
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    integer, dimension(nTurbines), intent(in) :: sorted_x_idx
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, TI, wind_speed, z_ref, z_0, shear_exp, opt_exp_fac
    real(dp), dimension(nRotorPoints), intent(in) :: RotorPointsY, RotorPointsZ
    real(dp), dimension(nCtPoints), intent(in) :: ct_curve_wind_speed, ct_curve_ct

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw, TIturbs, Ct_local, ky_local, kz_local
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset, k_star
    real(dp) :: x, deltav, deltaz, sigmay_dp, sigmaz_dp, deltax0_dp, deficit_sum
    real(dp) :: tol, discontinuity_point, TI_area_ratio 
    real(dp) :: TI_area_ratio_tmp, TI_dst_tmp, TI_ust_tmp, rpts
    real(dp) :: LocalRotorPointY, LocalRotorPointZ, point_velocity, point_z, point_velocity_with_shear
    Integer :: u, d, turb, turbI, p
    real(dp), parameter :: pi = 3.141592653589793_dp

    ! model out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity

    intrinsic sin, cos, atan, max, sqrt, log
    
    ! bastankhah and porte agel 2016 define yaw to be positive clockwise, this is reversed
    yaw = - yawDeg*pi/180.0_dp
    
    ! set tolerance for location checks
    tol = 0.1_dp
    
    ! initialize wind turbine velocities to 0.0
    wtVelocity = 0.0_dp
    
    ! initialize TI of all turbines to free-stream value
    !print *, "start TIturbs: ", TIturbs
    TIturbs = TI
    
    
    ! initialize the local wake factors
    ky_local(:) = ky
    kz_local(:) = kz
    Ct_local(:) = Ct


    !print *, 'wake model version: ', wake_model_version
    
    !print *, "ky_local: ", ky_local
    !print *, "kz_local: ", kz_local
    !print *, "TIturbs init: ", TIturbs

    do, d=1, nTurbines
    
        ! get index of downstream turbine
        turbI = sorted_x_idx(d) + 1
        
        do, p=1, nRotorPoints
        
            ! initialize the TI_area_ratio to 0.0 for each turbine
            TI_area_ratio = 0.0_dp
    
            ! initialize deficit summation term to zero
            deficit_sum = 0.0_dp
        
            ! scale rotor sample point coordinate by rotor diameter (in rotor hub ref. frame)
            LocalRotorPointY = RotorPointsY(p)*0.5_dp*rotorDiameter(turbI)
            LocalRotorPointZ = RotorPointsZ(p)*0.5_dp*rotorDiameter(turbI)
!             !print *, "rotorDiameter after local rotor points", rotorDiameter
!             !print *, "local rotor points Y,Z: ", LocalRotorPointY, LocalRotorPointZ
        
            do, u=1, nTurbines ! at turbineX-locations
            
                ! get index of upstream turbine
                turb = sorted_x_idx(u) + 1
                
                ! skip this loop if turb = turbI (turbines impact on itself)
                if (turb .eq. turbI) cycle
            
                ! downstream distance between upstream turbine and point
                x = turbineXw(turbI) - turbineXw(turb) + LocalRotorPointY*sin(yaw(turbI))
            
                ! set this iterations velocity deficit to 0
                deltav = 0.0_dp
                
                ! check turbine relative locations
                if (x > (0.0_dp + tol)) then
                
                    !print *, "rotorDiameter before x0 ", rotorDiameter
                
                    ! determine the onset location of far wake
                    call x0_func(rotorDiameter(turb), yaw(turb), Ct_local(turb), alpha, & 
                                & TIturbs(turb), beta, x0)
!                     call x0_func(rotorDiameter(turb), yaw(turb), Ct(turb), alpha, & 
!                                 & TI, beta, x0)
        
                    ! downstream distance from far wake onset to downstream turbine
                    deltax0 = x - x0
                
                    ! calculate wake spreading parameter at each turbine if desired
                    if (calc_k_star .eqv. .true.) then
                        call k_star_func(TIturbs(turb), k_star)
                        ky_local(turb) = k_star
                        kz_local(turb) = k_star
                    end if
                    
                    !print *, "ky_local ", ky_local
                    !print *, "deltax0 ", deltax0
                    !print *, "turbineZ ", turbineZ
                    !print *, "rotorDiameter after x0 ", rotorDiameter
                    !print *, "Ct ", Ct
                    !print *, "yaw ", yaw

                    ! determine the initial wake angle at the onset of far wake
                    call theta_c_0_func(yaw(turb), Ct_local(turb), theta_c_0)
                    !print *, "theta_c_0 ", theta_c_0
                    ! horizontal spread
                    call sigmay_func(ky_local(turb), deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                    !print *, "sigmay ", sigmay
                    !print *, "rotorDiameter after sigmay", rotorDiameter
                    ! vertical spread
                    call sigmaz_func(kz_local(turb), deltax0, rotorDiameter(turb), sigmaz)
                    !print *, "sigmaz ", sigmaz
                    !print *, "rotorDiameter after sigmaz ", rotorDiameter
                    ! horizontal cross-wind wake displacement from hub
                    call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                         & ky_local(turb), kz_local(turb), Ct_local(turb), sigmay, sigmaz, wake_offset)
                                         
                                         
                    !print *, "wake_offset ", wake_offset                 
                    ! cross wind distance from downstream point location to wake center
                    deltay = LocalRotorPointY*cos(yaw(turbI)) + turbineYw(turbI) - (turbineYw(turb) + wake_offset)
            
                    ! cross wind distance from hub height to height of point of interest
                    deltaz = LocalRotorPointZ + turbineZ(turbI) - turbineZ(turb)
                    
                    !print *, "dx, dy, dz: ", x, deltay, deltaz
                    !print *, "local y,z : ", LocalRotorPointY, LocalRotorPointZ, turb, turbI, p
                    !print *, deltaz, deltay
                    ! far wake region
                    
                    ! find the final point where the original model is undefined
                    call discontinuity_point_func(x0, rotorDiameter(turb), ky_local(turb), &
                                                 & kz_local(turb), yaw(turb), Ct_local(turb), & 
                                                 & discontinuity_point)
                    
                    if (x > discontinuity_point) then
                    
                        !print *, x
    
                        ! velocity difference in the wake
                        call deltav_func(deltay, deltaz, Ct_local(turb), yaw(turb), &
                                        & sigmay, sigmaz, rotorDiameter(turb), & 
                                        & wake_model_version, kz_local(turb), x, &
                                        & opt_exp_fac, deltav)  
                        !print *, "rotorDiameter after far deltav ", rotorDiameter
                    ! near wake region (linearized)
                    else
                        
                        ! determine distance from discontinuity point to far wake onset
                        deltax0_dp = discontinuity_point - x0
                
                        ! horizontal spread at far wake onset
                        call sigmay_func(ky_local(turb), deltax0_dp, rotorDiameter(turb), yaw(turb), sigmay_dp)
                
                        ! vertical spread at far wake onset
                        call sigmaz_func(kz_local(turb), deltax0_dp, rotorDiameter(turb), sigmaz_dp)

                       !  print *, "inputs in parent: ", deltay, deltaz, Ct(turb), yaw(turb), sigmay_dp, sigmaz_dp, &
!                                          & rotorDiameter(turb), x, discontinuity_point, sigmay_dp, sigmaz_dp, &
!                                          & wake_model_version, kz_local, x0, &
!                                          & opt_exp_fac

                        ! velocity deficit in the nearwake (linear model)
                        call deltav_near_wake_lin_func(deltay, deltaz, &
                                         & Ct_local(turb), yaw(turb), sigmay_dp, sigmaz_dp, & 
                                         & rotorDiameter(turb), x, discontinuity_point, sigmay_dp, sigmaz_dp, & 
                                         & wake_model_version, kz_local(turb), x0, & 
                                         & opt_exp_fac, deltav)
                                         
                        !print *, "rotorDiameter after deltav near ", rotorDiameter
                    end if
                
                    ! combine deficits according to selected method wake combination method
                    call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                               wake_combination_method, deficit_sum)
                
                    if ((x > 0.0_dp) .and. (TI_calculation_method > 0)) then
                        !print *, "turbI, turb: ", turbI, turb
                        ! calculate TI value at each turbine
!                         print *, "turb, turbI: ", turb, turbI
                        
                        ! save ti_area_ratio and ti_dst to new memory locations to avoid 
                        ! aliasing during differentiation
                        TI_area_ratio_tmp = TI_area_ratio
                        TI_dst_tmp = TIturbs(turbI)
                        TI_ust_tmp = TIturbs(turb)
                        
                        call added_ti_func(TI, Ct_local(turb), x, ky_local(turb), rotorDiameter(turb), & 
                                           & rotorDiameter(turbI), deltay, turbineZ(turb), &
                                           & turbineZ(turbI), TI_ust_tmp, &
                                           & TI_calculation_method, TI_area_ratio_tmp, &
                                           & TI_dst_tmp, TI_area_ratio, TIturbs(turbI))
                                           
                        !print *, "rotorDiameter after TI calcs", rotorDiameter
                    end if
                    
!                     !print *, "deficit_sum, turbI, p, turb: ", deficit_sum, turbI, p, turb
                
                end if
            
            end do
            
            ! print *, deficit_sum
            
            ! find velocity at point p due to the wake of turbine turb
            point_velocity = wind_speed - deficit_sum
            
            !print *, "point velocity, deficit_sum, turbI, p: ", point_velocity, deficit_sum, turbI, p    
        
            ! put sample point height in global reference frame
            point_z = LocalRotorPointZ + turbineZ(turbI)
        
            !print *, "point_z, turbI, p: ", point_z, turbI, p    
            ! adjust sample point velocity for shear
            call wind_shear_func(point_z, point_velocity, z_ref, z_0, shear_exp, point_velocity_with_shear)
            !print *, "v, vs, x, turb, turbI, p: ", point_velocity, point_velocity_with_shear, x, turb, turbI, p
            ! add sample point velocity to turbine velocity to be averaged later
            wtVelocity(turbI) = wtVelocity(turbI) + point_velocity_with_shear
        
        end do
    
        ! final velocity calculation for turbine turbI (average equally across all points)
        rpts = REAL(nRotorPoints, dp)
!         print *, rpts, nRotorPoints, wtVelocity(turbI), wtVelocity(turbI)/rpts, wtVelocity(turbI)/nRotorPoints
!         STOP 1
        wtVelocity(turbI) = wtVelocity(turbI)/rpts
!         print *, wtVelocity(turbI)
        if (use_ct_curve) then
            call interpolation(nCtPoints, interp_type, ct_curve_wind_speed, ct_curve_ct, & 
                              & wtVelocity(turbI), Ct_local(turbI))
        end if
    
    end do
         
   !!  print TIturbs values to a file
!     if (print_ti) then
!         open(unit=2, file="TIturbs_tmp.txt")
!         do, turb=1, nTurbines 
!             write(2,*) TIturbs(turb)
!         end do
!         close(2)
!     end if 
    
    !print *, "TIturbs: ", TIturbs
    !print *, wtVelocity

    !! make sure turbine inflow velocity is non-negative
!             if (wtVelocity(turbI) .lt. 0.0_dp) then 
!                 wtVelocity(turbI) = 0.0_dp
!             end if
    !print *, "fortran"

end subroutine porteagel_analyze

! implementation of the Bastankhah and Porte Agel (BPA) wake model for visualization
subroutine porteagel_visualize(nTurbines, nSamples, nRotorPoints, nCtPoints, turbineXw, &
                             sorted_x_idx, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, TI, RotorPointsY, RotorPointsZ, & 
                             z_ref, z_0, shear_exp, velX, velY, velZ, &
                             wake_combination_method, TI_calculation_method, &
                             calc_k_star, opt_exp_fac, wake_model_version, interp_type, &
                             use_ct_curve, ct_curve_wind_speed, ct_curve_ct, wsArray)
                             
    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        Ct yawDeg

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nSamples, nRotorPoints, nCtPoints
    integer, intent(in) :: wake_combination_method, TI_calculation_method, & 
                        &  wake_model_version, interp_type
    logical, intent(in) :: calc_k_star, use_ct_curve
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    integer, dimension(nTurbines), intent(in) :: sorted_x_idx
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, TI, wind_speed, z_ref, z_0, shear_exp, opt_exp_fac
    real(dp), dimension(nRotorPoints), intent(in) :: RotorPointsY, RotorPointsZ
    real(dp), dimension(nCtPoints), intent(in) :: ct_curve_wind_speed, ct_curve_ct
    real(dp), dimension(nSamples), intent(in) :: velX, velY, velZ

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw, TIturbs, wtVelocity
    real(dp) :: TI_area_ratio, TI_area_ratio_tmp, TI_dst_tmp, TI_ust_tmp
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset, discontinuity_point
    real(dp) :: x, deltav, deltaz, sigmay_dp, sigmaz_dp, deltax0_dp, deficit_sum, tol, k_star
    real(dp) :: LocalRotorPointY, LocalRotorPointZ, point_velocity, point_z, point_velocity_with_shear
    real(dp), dimension(nTurbines) :: ky_local, kz_local, Ct_local
    Integer :: u, d, turb, turbI, loc, p
    real(dp), parameter :: pi = 3.141592653589793_dp

    ! model out
    real(dp), dimension(nSamples), intent(out) :: wsArray

    intrinsic sin, cos, atan, max, sqrt, log
    
    ! bastankhah and porte agel 2016 define yaw to be positive clockwise, this is reversed
    yaw = - yawDeg*pi/180.0_dp
    
    ! set tolerance for location checks
    tol = 0.1_dp

    ! initialize location velocities to free stream
    wsArray = wind_speed
    
    ! initialize wind turbine velocities to 0.0
    wtVelocity = 0.0_dp
    
    ! initialize TI of all turbines to free-stream values
    TIturbs = TI
    
    ! initialize local wake values to free-stream values
    ky_local(:) = ky
    kz_local(:) = kz
    Ct_local(:) = Ct
    
    !print *, "entering turbine calculation loops"
    
    do, d=1, nTurbines
    
        ! get index of downstream turbine
        turbI = sorted_x_idx(d) + 1
        
        do, p=1, nRotorPoints
        
            ! initialize the TI_area_ratio to 0.0 for each turbine
            TI_area_ratio = 0.0_dp
    
            ! initialize deficit summation term to zero
            deficit_sum = 0.0_dp
        
            ! scale rotor sample point coordinate by rotor diameter (in rotor hub ref. frame)
            LocalRotorPointY = RotorPointsY(p)*0.5_dp*rotorDiameter(turbI)
            LocalRotorPointZ = RotorPointsZ(p)*0.5_dp*rotorDiameter(turbI)
            
            do, u=1, nTurbines ! at turbineX-locations
            
                ! get index of upstream turbine
                turb = sorted_x_idx(u) + 1
                
                ! skip this loop if turb = turbI (turbines impact on itself)
                if (turb .eq. turbI) cycle
        
                ! downstream distance between turbines, adjust for downstream turbine yaw
                x = turbineXw(turbI) - turbineXw(turb) + LocalRotorPointY*sin(yaw(turbI))
            
                ! set this iterations velocity deficit to 0
                deltav = 0.0_dp
                
                ! check turbine relative locations
                if (x > (0.0_dp + tol)) then
                
                    ! determine the onset location of far wake
                    call x0_func(rotorDiameter(turb), yaw(turb), Ct_local(turb), alpha, & 
                                & TIturbs(turb), beta, x0)
        
                    ! downstream distance from far wake onset to downstream turbine
                    deltax0 = x - x0
                
                    ! calculate wake spreading parameter at each turbine if desired
                    if (calc_k_star .eqv. .true.) then
                        call k_star_func(TIturbs(turb), k_star)
                        ky_local(turb) = k_star
                        kz_local(turb) = k_star
                    end if

                    ! determine the initial wake angle at the onset of far wake
                    call theta_c_0_func(yaw(turb), Ct_local(turb), theta_c_0)
        
                    ! horizontal spread
                    call sigmay_func(ky_local(turb), deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                    
                    ! vertical spread
                    call sigmaz_func(kz_local(turb), deltax0, rotorDiameter(turb), sigmaz)
                    
                    ! horizontal cross-wind wake displacement from hub
                    call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                         & ky_local(turb), kz_local(turb), Ct_local(turb), sigmay, sigmaz, wake_offset)
                              
                    ! cross wind distance from downstream point location to wake center
                    deltay = LocalRotorPointY*cos(yaw(turbI)) + turbineYw(turbI) - (turbineYw(turb) + wake_offset)
            
                    ! cross wind distance from hub height to height of point of interest
                    deltaz = LocalRotorPointZ + turbineZ(turbI) - turbineZ(turb)
         
                    ! find the final point where the original model is undefined
                    call discontinuity_point_func(x0, rotorDiameter(turb), ky_local(turb), &
                                                 & kz_local(turb), yaw(turb), Ct_local(turb), & 
                                                 & discontinuity_point)
                                                 
                    ! far wake region
                    if (x > discontinuity_point) then
    
                        ! velocity difference in the wake
                        call deltav_func(deltay, deltaz, Ct_local(turb), yaw(turb), &
                                        & sigmay, sigmaz, rotorDiameter(turb), & 
                                        & wake_model_version, kz_local(turb), x, &
                                        & opt_exp_fac, deltav)  
                                        
                    ! near wake region (linearized)
                    else
                        
                        ! determine distance from discontinuity point to far wake onset
                        deltax0_dp = discontinuity_point - x0
                
                        ! horizontal spread at discontinuity point
                        call sigmay_func(ky_local(turb), deltax0_dp, rotorDiameter(turb), yaw(turb), sigmay_dp)
                
                        ! vertical spread at discontinuity point
                        call sigmaz_func(kz_local(turb), deltax0_dp, rotorDiameter(turb), sigmaz_dp)

                        ! print *, "inputs in parent: ", deltay, deltaz, Ct(turb), yaw(turb), sigmay_dp, sigmaz_dp, &
!                                          & rotorDiameter(turb), x, discontinuity_point, sigmay_dp, sigmaz_dp, &
!                                          & wake_model_version, kz_local, x0, &
!                                          & opt_exp_fac

                        ! velocity deficit in the nearwake (linear model)
                        call deltav_near_wake_lin_func(deltay, deltaz, &
                                         & Ct_local(turb), yaw(turb), sigmay_dp, sigmaz_dp, & 
                                         & rotorDiameter(turb), x, discontinuity_point, sigmay_dp, sigmaz_dp, & 
                                         & wake_model_version, kz_local(turb), x0, & 
                                         & opt_exp_fac, deltav)
                                         
              
                    end if
                
                    ! combine deficits according to selected method wake combination method
                    call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                               wake_combination_method, deficit_sum)
                
                    if ((x > 0.0_dp) .and. (TI_calculation_method > 0)) then
                    
                        ! save ti_area_ratio value to new memory location to avoid aliasing
                        TI_area_ratio_tmp = TI_area_ratio
                        TI_dst_tmp = TIturbs(turbI)
                        TI_ust_tmp = TIturbs(turb)
                
                        ! calculate TI value at each turbine
                        call added_ti_func(TI, Ct_local(turb), x, ky_local(turb), rotorDiameter(turb), & 
                                           & rotorDiameter(turbI), deltay, turbineZ(turb), &
                                           & turbineZ(turbI), TI_ust_tmp, &
                                           & TI_calculation_method, TI_area_ratio_tmp, & 
                                           & TI_dst_tmp, TI_area_ratio, &
                                           & TIturbs(turbI))
                    end if
                
                end if
                
            end do
            
            ! find velocity at point p due to the wake of turbine turb
            point_velocity = wind_speed - deficit_sum
            
            ! put sample point height in global reference frame
            point_z = LocalRotorPointZ + turbineZ(turbI)
            
            ! adjust sample point velocity for shear
            call wind_shear_func(point_z, point_velocity, z_ref, z_0, shear_exp, point_velocity_with_shear)
            
            ! add sample point velocity to turbine velocity to be averaged later
            wtVelocity(turbI) = wtVelocity(turbI) + point_velocity_with_shear
        
        end do
    
        ! final velocity calculation for turbine turbI (average equally across all points)
        wtVelocity(turbI) = wtVelocity(turbI)/nRotorPoints
        
        if (use_ct_curve) then
            call interpolation(nCtPoints, interp_type, ct_curve_wind_speed, ct_curve_ct, & 
                              & wtVelocity(turbI), Ct_local(turbI))
        end if
       
    end do

    do, loc=1, nSamples
         
        ! set combined velocity deficit to zero for loc
        deficit_sum = 0.0_dp

        do, turb=1, nTurbines ! at turbineX-locations

            ! set velocity deficit at loc due to turb to zero
            deltav = 0.0_dp

            ! downstream distance between turbines
            x = velX(loc) - turbineXw(turb)
            
            if (x > 0.0_dp) then
            
                ! determine the onset location of far wake
                call x0_func(rotorDiameter(turb), yaw(turb), Ct_local(turb), alpha, TIturbs(turb), &
                            & beta, x0)
                
                ! downstream distance from far wake onset to downstream turbine
                deltax0 = x - x0
        
                ! determine the initial wake angle at the onset of far wake
                call theta_c_0_func(yaw(turb), Ct_local(turb), theta_c_0)
        
                ! horizontal spread
                call sigmay_func(ky_local(turb), deltax0, rotorDiameter(turb), yaw(turb), sigmay)
            
                ! vertical spread
                call sigmaz_func(kz_local(turb), deltax0, rotorDiameter(turb), sigmaz)
            
                ! horizontal cross-wind wake displacement from hub
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky_local(turb), kz_local(turb), Ct_local(turb), sigmay, &
                                     & sigmaz, wake_offset)
                                 
                ! horizontal distance from downstream hub location to wake center
                deltay = velY(loc) - (turbineYw(turb) + wake_offset)
            
                ! vertical distance from downstream hub location to wake center
                deltaz = velZ(loc) - turbineZ(turb)
                                
                ! find the final point where the original model is undefined
                call discontinuity_point_func(x0, rotorDiameter(turb), ky_local(turb), kz_local(turb), yaw(turb), Ct_local(turb), & 
                                             & discontinuity_point)
                                             
                ! far wake region
                if (x > discontinuity_point) then
                
                    ! velocity difference in the wake
                    call deltav_func(deltay, deltaz, Ct_local(turb), yaw(turb),  &
                                    & sigmay, sigmaz, rotorDiameter(turb), & 
                                    & wake_model_version, kz_local(turb), x, & 
                                    & opt_exp_fac, deltav)
                                    
                ! near wake region (linearized)
                else
                    
                    ! determine distance from discontinuity point to far wake onset
                    deltax0_dp = discontinuity_point - x0
                
                    ! horizontal spread at far wake onset
                    call sigmay_func(ky_local(turb), deltax0_dp, rotorDiameter(turb), yaw(turb), &
                                    & sigmay_dp)
                                    
                    ! vertical spread at far wake onset
                    call sigmaz_func(kz_local(turb), deltax0_dp, rotorDiameter(turb), sigmaz_dp)
                    
                    ! velocity deficit in the nearwake (linear model)
                    call deltav_near_wake_lin_func(deltay, deltaz,        &
                                     & Ct_local(turb), yaw(turb), sigmay_dp, sigmaz_dp,           & 
                                     & rotorDiameter(turb), x, discontinuity_point, & 
                                     & sigmay_dp, sigmaz_dp,    & 
                                     & wake_model_version, kz_local(turb), x0, &
                                     & opt_exp_fac, deltav)

                end if
            
                ! combine deficits according to selected method wake combination method
                call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                           wake_combination_method, deficit_sum) 
                                           
            end if
            
        end do
        
        ! find velocity at point p due to the wake of turbine turb
        point_velocity = wind_speed - deficit_sum
        
        ! put sample point height in global reference frame
        point_z = velZ(loc)
        
        ! adjust sample point velocity for shear
        call wind_shear_func(point_z, point_velocity, z_ref, z_0, shear_exp, point_velocity_with_shear)

            
        ! final velocity calculation for location loc
        wsArray(loc) = point_velocity_with_shear
 
    end do

end subroutine porteagel_visualize


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
subroutine sigmay_func(ky, deltax0, rotor_diameter, yaw, sigmay)
    
    implicit none

    ! define precision to be the standard for a double precision on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: ky, deltax0, rotor_diameter, yaw

    ! out
    real(dp), intent(out) :: sigmay

    intrinsic cos, sqrt
    
    ! horizontal spread
    sigmay = rotor_diameter * (ky * deltax0 / rotor_diameter + cos(yaw) / sqrt(8.0_dp))
    
end subroutine sigmay_func
    
    
! calculates the vertical spread of the wake at a given distance from the onset of far 
! wake condition
subroutine sigmaz_func(kz, deltax0, rotor_diameter, sigmaz)
    
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: kz, deltax0, rotor_diameter

    ! out
    real(dp), intent(out) :: sigmaz

    ! load necessary intrinsic functions
    intrinsic sqrt
    
    ! vertical spread
    sigmaz = rotor_diameter * (kz * deltax0 / rotor_diameter + 1.0_dp / sqrt(8.0_dp))
    
end subroutine sigmaz_func


! calculates the horizontal distance from the wake center to the hub of the turbine making
! the wake
subroutine wake_offset_func(rotor_diameter, theta_c_0, x0, yaw, ky, kz, Ct, sigmay, &
                            & sigmaz, wake_offset)
                            
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: rotor_diameter, theta_c_0, x0, yaw, ky, kz, Ct, sigmay
    real(dp), intent(in) :: sigmaz

    ! out
    real(dp), intent(out) :: wake_offset

    intrinsic cos, sqrt, log
                            
    ! horizontal cross-wind wake displacement from hub
    wake_offset = rotor_diameter * (                                           &
                  theta_c_0 * x0 / rotor_diameter +                            &
                  (theta_c_0 / 14.7_dp) * sqrt(cos(yaw) / (ky * kz * Ct)) *    &
                  (2.9_dp + 1.3_dp * sqrt(1.0_dp - Ct) - Ct) *                 &
                  log(                                                         &
                    ((1.6_dp + sqrt(Ct)) *                                     &
                     (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /                 &
                                    (cos(yaw) * rotor_diameter ** 2))          &
                      - sqrt(Ct))) /                                           &
                    ((1.6_dp - sqrt(Ct)) *                                     &
                     (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /                 &
                                    (cos(yaw) * rotor_diameter ** 2))          &
                      + sqrt(Ct)))                                             &
                  )                                                            &
    )
end subroutine wake_offset_func


! calculates the velocity difference between hub velocity and free stream for a given wake
! for use in the far wake region
subroutine deltav_func(deltay, deltaz, Ct, yaw, sigmay, sigmaz, & 
                      & rotor_diameter_ust, version, k, deltax, opt_exp_fac, deltav) 
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter_ust, opt_exp_fac
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
            ((deltaz/(opt_exp_fac*rotor_diameter_ust))**2 + (deltay/(opt_exp_fac*rotor_diameter_ust))**2))           &
        )
       ! print *, "deltav 2014 = ", deltav
    else if (version == 2016) then
        ! velocity difference in the wake at each sample point
        deltav = (                                                                    &
            (1.0_dp - sqrt(1.0_dp - Ct *                                                         &
                           cos(yaw) / (8.0_dp * sigmay * sigmaz / (rotor_diameter_ust ** 2)))) *     &
            exp(-0.5_dp * (deltay / (opt_exp_fac*sigmay)) ** 2) * exp(-0.5_dp * (deltaz / (opt_exp_fac*sigmaz)) ** 2)&
        )
    else
        print *, "Invalid Bastankhah and Porte Agel model version. Must be 2014 or 2016. ", version, " was given."
        stop 1
    end if 
    
    !print *, "rotor_diameter in deltav exit", rotor_diameter_ust

end subroutine deltav_func


! calculates the velocity difference between hub velocity and free stream for a given wake
! for use in the near wake region only
subroutine deltav_near_wake_lin_func(deltay, deltaz, Ct, yaw,  &
                                 & sigmay, sigmaz, rotor_diameter_ust, x, &
                                 & discontinuity_point, sigmay0, sigmaz0, version, k, &
                                 & deltax0_dp, opt_exp_fac, deltav) 
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter_ust, opt_exp_fac
    real(dp), intent(in) :: x, discontinuity_point, sigmay0, sigmaz0
    real(dp), intent(in) :: k, deltax0_dp    ! only for 2014 version
    integer, intent(in) :: version
    
    ! local
    real(dp) :: deltav0m, deltavr
    real(dp) :: beta_2014, epsilon_2014 ! only for 2014 version

    ! out
    real(dp), intent(out) :: deltav

    ! load intrinsic functions
    intrinsic cos, sqrt, exp

   !  print *, 'wake model version in deltav near wake: ', version
!     print *, "inputs: ", deltay, deltaz, Ct, yaw,  &
!                                  & sigmay, sigmaz, rotor_diameter_ust, x, &
!                                  & discontinuity_point, sigmay0, sigmaz0, version, k, &
!                                  & deltax0_dp, opt_exp_fac
    if (version == 2014) then
        if (yaw > 0.0_dp) then
            print *, "model version 2014 may only be used when yaw=0"
            stop 1
        end if
        beta_2014 = 0.5_dp*(1.0_dp + sqrt(1.0_dp - Ct))/sqrt(1.0_dp - Ct)
        epsilon_2014 = 0.2_dp*sqrt(beta_2014)
        
        ! magnitude term of gaussian at x0
        deltav0m = (1.0_dp - sqrt(1.0_dp - Ct                                            &
                           / (8.0_dp * (k*deltax0_dp/rotor_diameter_ust+epsilon_2014)**2)))
        
        ! initialize the gaussian magnitude term at the rotor for the linear interpolation
        deltavr = deltav0m
        
        ! linearized gaussian magnitude term for near wake
        deltav = (                                                                       &
             (((deltav0m - deltavr)/discontinuity_point) * x + deltavr) *                &
            exp((-1.0_dp/(2.0_dp*(k*deltax0_dp/rotor_diameter_ust + epsilon_2014)**2))*      & 
            ((deltaz/(opt_exp_fac*rotor_diameter_ust))**2 + (deltay/(opt_exp_fac*rotor_diameter_ust))**2))           &
        )
    else if (version == 2016) then

        ! magnitude term of gaussian at x0
        deltav0m = (                                         &
                    (1.0_dp - sqrt(1.0_dp - Ct *                          &
                    cos(yaw) / (8.0_dp * sigmay0 * sigmaz0 /              &
                                                (rotor_diameter_ust ** 2)))))
        ! initialize the gaussian magnitude term at the rotor for the linear interpolation
        deltavr = deltav0m

        ! linearized gaussian magnitude term for near wake
        deltav = (((deltav0m - deltavr)/discontinuity_point) * x + deltavr) *       &
            exp(-0.5_dp * (deltay / (opt_exp_fac*sigmay)) ** 2) *                                 &
            exp(-0.5_dp * (deltaz / (opt_exp_fac*sigmaz)) ** 2)
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
                                 wake_combination_method, deficit_sum)
                                 
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
    real(dp), intent(in) :: wind_speed, turb_inflow, deltav
    integer, intent(in) :: wake_combination_method
    
    ! out    
    real(dp), intent(inout) :: deficit_sum
    
    ! intrinsic functions
    intrinsic sqrt
    
    ! freestream linear superposition (Lissaman 1979)
    if (wake_combination_method == 0) then
        deficit_sum = deficit_sum + wind_speed*deltav

    ! local velocity linear superposition (Niayifar and Porte Agel 2015, 2016)
    else if (wake_combination_method == 1) then
        deficit_sum = deficit_sum + turb_inflow*deltav
        !print *, "here"
        
    ! sum of squares freestream superposition (Katic et al. 1986)
    else if (wake_combination_method == 2) then 
        deficit_sum = sqrt(deficit_sum**2 + (wind_speed*deltav)**2)
    
    ! sum of squares local velocity superposition (Voutsinas 1990)
    else if (wake_combination_method == 3) then
        deficit_sum = sqrt(deficit_sum**2 + (turb_inflow*deltav)**2)
    
    ! wake combination method error
    else
        print *, "Invalid wake combination method. Must be one of [0,1,2,3]."
        stop 1
    end if                       
    
end subroutine wake_combination_func

! combines wakes using various methods
subroutine added_ti_func(TI, Ct_ust, x, k_star_ust, rotor_diameter_ust, rotor_diameter_dst, & 
                        & deltay, wake_height, turbine_height, TI_ust, &
                        & TI_calculation_method, TI_area_ratio_in, TI_dst_in, TI_area_ratio, TI_dst)
                                 
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: Ct_ust, x, k_star_ust, rotor_diameter_ust, rotor_diameter_dst
    real(dp), intent(in) :: deltay, wake_height, turbine_height, TI_ust, TI
    real(dp), intent(in) :: TI_area_ratio_in, TI_dst_in
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
        call smooth_max(TI_dst_in, TI_tmp, TI_dst)
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
        TI_tmp = sqrt(TI**2.0_dp + (TI_added*(wake_overlap/rotor_area_dst))**2.0_dp)
        
        ! Check if this is the max and use it if it is
        if (TI_area_ratio_tmp > TI_area_ratio_in) then
!            print *, "ti_area_ratio_tmp > ti_area_ratio"
           TI_dst = TI_tmp
           TI_area_ratio = TI_area_ratio_tmp
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
                            
        ! Calculate the turbulence added to the inflow of the downstream turbine by the 
        ! wake of the upstream turbine
        TI_added = 0.73_dp*(axial_induction_ust**0.8325_dp)*(TI_ust**0.0325_dp)* & 
                    ((x/rotor_diameter_ust)**(-0.32_dp))
        
        ! Calculate the total turbulence intensity at the downstream turbine based on 
        ! current upstream turbine
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2_dp
        TI_area_ratio_tmp = TI_added*(wake_overlap/rotor_area_dst)
        TI_tmp = sqrt(TI**2.0_dp + (TI_added*(wake_overlap/rotor_area_dst))**2.0_dp)
        
        ! Check if this is the max and use it if it is
!         TI_dst_in = TI_dst
        call smooth_max(TI_dst_in, TI_tmp, TI_dst)
     
    !print *, "sigma: ", sigma
    ! TODO add other TI calculation methods
    
        
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

subroutine smooth_max(x, y, g)

    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: x, y
    
    ! out
    real(dp), intent(out) :: g
    
    ! local
    real(dp) :: s
    
    intrinsic log, exp
    
    s = 100.0_dp
    
!     g = (log(exp(s*x) + exp(s*y)))/s
!     print *, "g1 = ", g
    
    g = (x*exp(s*x)+y*exp(s*y))/(exp(s*x)+exp(s*y))
!     print *, "g2 = ", g
    
!     print *, "g is ", g
end subroutine smooth_max

subroutine interpolation(nPoints, interp_type, x, y, xval, yval)

    implicit none
    
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nPoints, interp_type
    real(dp), dimension(nPoints), intent(in) :: x, y
    real(dp), intent(in) :: xval
    
    ! local
    integer :: idx
    real(dp) :: x0, x1, y0, dy0, y1, dy1
    
    ! out
    real(dp), intent(out) :: yval
    
!     print *, "in interpolation"
    
    if ((xval < x(1)) .or. (xval > x(nPoints))) then
        print *, "interpolation point is out of bounds"
!         STOP 1
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
                dy0 = 0.0_dp
            else
                dy0 = (y(idx) - y(idx-1))/(x(idx) - x(idx-1))
            end if
    
            ! approximate derivative at the right end of interval
            if (idx >= nPoints-1) then
                dy1 = 0.0_dp
            else
                dy1 = (y(idx+2) - y(idx+1))/(x(idx+2) - x(idx+1))
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
