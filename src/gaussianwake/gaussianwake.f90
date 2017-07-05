! Implementation of the Bastankhah and Porte Agel gaussian-shaped wind turbine wake 
! model (2016) with various farm modeling (TI and wake combination) methods included
! Created by Jared J. Thomas, 2017.
! FLight Optimization and Wind Laboratory (FLOW Lab)
! Brigham Young University

! implementation of the Bastankhah and Porte Agel (BPA) wake model for analysis
subroutine porteagel_analyze(nTurbines, nRotorPoints, turbineXw, sorted_x_idx, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, TI, RotorPointsY, RotorPointsZ, &
                             z_ref, z_0, shear_exp, wake_combination_method, &
                             TI_calculation_method, calc_k_star, wtVelocity)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        Ct yawDeg

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nRotorPoints
    integer, intent(in) :: wake_combination_method, TI_calculation_method
    logical, intent(in) :: calc_k_star
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    integer, dimension(nTurbines), intent(in) :: sorted_x_idx
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, TI, wind_speed, z_ref, z_0, shear_exp
    real(dp), dimension(nRotorPoints), intent(in) :: RotorPointsY, RotorPointsZ

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw, TIturbs, k_star
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset
    real(dp) :: x, deltav, deltav0m, deltaz, sigmay_dp, sigmaz_dp, deltax0_dp, deficit_sum
    real(dp) :: ky_local, kz_local, tol, discontinuity_point
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
    !print *, "initialized TIturbs: ", TIturbs
    ky_local = ky
    kz_local = kz

    do, d=1, nTurbines
    
        ! get index of downstream turbine
        turbI = sorted_x_idx(d) + 1
        
        do, p=1, nRotorPoints
    
            ! initialize deficit summation term to zero
            deficit_sum = 0.0_dp
        
            ! scale rotor sample point coordinate by rotor diameter (in rotor hub ref. frame)
            LocalRotorPointY = RotorPointsY(p)*0.5_dp*rotorDiameter(turbI)
            LocalRotorPointZ = RotorPointsZ(p)*0.5_dp*rotorDiameter(turbI)
!             print *, "rotorDiameter after local rotor points", rotorDiameter
!             print *, "local rotor points Y,Z: ", LocalRotorPointY, LocalRotorPointZ
        
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
                    call x0_func(rotorDiameter(turb), yaw(turb), Ct(turb), alpha, & 
                                & TIturbs(turb), beta, x0)
        
                    ! downstream distance from far wake onset to downstream turbine
                    deltax0 = x - x0
                
                    ! calculate wake spreading parameter at each turbine if desired
                    if (calc_k_star .eqv. .true.) then
                        call k_star_func(TIturbs(turb), k_star(turb))
                        ky_local = k_star(turb)
                        kz_local = k_star(turb)
                    end if
                    
                    !print *, "ky_local ", ky_local
                    !print *, "deltax0 ", deltax0
                    !print *, "turbineZ ", turbineZ
                    !print *, "rotorDiameter after x0 ", rotorDiameter
                    !print *, "Ct ", Ct
                    !print *, "yaw ", yaw

                    ! determine the initial wake angle at the onset of far wake
                    call theta_c_0_func(yaw(turb), Ct(turb), theta_c_0)
                    !print *, "theta_c_0 ", theta_c_0
                    ! horizontal spread
                    call sigmay_func(ky_local, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                    !print *, "sigmay ", sigmay
                    !print *, "rotorDiameter after sigmay", rotorDiameter
                    ! vertical spread
                    call sigmaz_func(kz_local, deltax0, rotorDiameter(turb), sigmaz)
                    !print *, "sigmaz ", sigmaz
                    !print *, "rotorDiameter after sigmaz ", rotorDiameter
                    ! horizontal cross-wind wake displacement from hub
                    call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                         & ky_local, kz_local, Ct(turb), sigmay, sigmaz, wake_offset)
                                         
                                         
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
                    call discontinuity_point_func(x0, rotorDiameter(turb), ky, kz, yaw, Ct, & 
                                                 & discontinuity_point)
                    
                    if (x > discontinuity_point) then
                    
                        print *, x
    
                        ! velocity difference in the wake
                        call deltav_func(deltay, deltaz, Ct(turb), yaw(turb), &
                                        & sigmay, sigmaz, rotorDiameter(turb), deltav)  
                        !print *, "rotorDiameter after far deltav ", rotorDiameter
                    ! near wake region (linearized)
                    else
                        
                        ! determine distance from discontinuity point to far wake onset
                        deltax0_dp = discontinuity_point - x0
                
                        ! horizontal spread at far wake onset
                        call sigmay_func(ky_local, deltax0_dp, rotorDiameter(turb), yaw(turb), sigmay_dp)
                        !print *, "rotorDiameter after sigmay0", rotorDiameter
                
                        ! vertical spread at far wake onset
                        call sigmaz_func(kz_local, deltax0_dp, rotorDiameter(turb), sigmaz_dp)
                        !print *, "rotorDiameter after sigmaz0", rotorDiameter
                        ! velocity deficit in the nearwake (linear model)
                        call deltav_near_wake_lin_func(deltay, deltaz, &
                                         & Ct(turb), yaw(turb), sigmay_dp, sigmaz_dp, & 
                                         & rotorDiameter(turb), x, x0, sigmay_dp, sigmaz_dp, & 
                                         & deltav)
                        !print *, "rotorDiameter after deltav near ", rotorDiameter
                    end if
                
                    ! combine deficits according to selected method wake combination method
                    call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                               wake_combination_method, deficit_sum)
                
                    if ((deltax0 > 0.0_dp) .and. (TI_calculation_method > 0)) then
                
                        ! calculate TI value at each turbine
                        call added_ti_func(TI, Ct, deltax0, k_star(turb), rotorDiameter(turb), & 
                                           & rotorDiameter(turbI), deltay, turbineZ(turb), &
                                           & turbineZ(turbI), TIturbs(turb), &
                                           & TI_calculation_method, &
                                           & TIturbs(turbI))
                        !print *, "rotorDiameter after TI calcs", rotorDiameter
                    end if
                    
!                     print *, "deficit_sum, turbI, p, turb: ", deficit_sum, turbI, p, turb
                
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
        wtVelocity(turbI) = wtVelocity(turbI)/nRotorPoints
    
    end do
    
    
    !print *, wtVelocity

    !! make sure turbine inflow velocity is non-negative
!             if (wtVelocity(turbI) .lt. 0.0_dp) then 
!                 wtVelocity(turbI) = 0.0_dp
!             end if
    !print *, "fortran"

end subroutine porteagel_analyze

! implementation of the Bastankhah and Porte Agel (BPA) wake model for visualization
subroutine porteagel_visualize(nTurbines, nSamples, nRotorPoints, turbineXw, sorted_x_idx, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, TI, RotorPointsY, RotorPointsZ, & 
                             z_ref, z_0, shear_exp, velX, velY, velZ, &
                             wake_combination_method, TI_calculation_method, &
                             calc_k_star, wsArray)
                             
    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        Ct yawDeg

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nSamples, nRotorPoints
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    integer, dimension(nTurbines), intent(in) :: sorted_x_idx
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, TI, wind_speed, z_ref, z_0, shear_exp
    real(dp), dimension(nRotorPoints), intent(in) :: RotorPointsY, RotorPointsZ
    real(dp), dimension(nSamples), intent(in) :: velX, velY, velZ
    integer, intent(in) :: wake_combination_method, TI_calculation_method
    logical, intent(in) :: calc_k_star

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw, TIturbs, k_star, wtVelocity
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset, discontinuity_point
    real(dp) :: x, deltav, deltaz, sigmay_dp, sigmaz_dp, deltax0_dp, deficit_sum, tol
    real(dp) :: LocalRotorPointY, LocalRotorPointZ, point_velocity, point_z, point_velocity_with_shear
    real(dp), dimension(nTurbines) :: ky_local, kz_local
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
    
    ! initialize local k values to free-stream values
    ky_local = ky
    kz_local = kz
    
    do, d=1, nTurbines
    
        ! get index of downstream turbine
        turbI = sorted_x_idx(d) + 1
        
        do, p=1, nRotorPoints
    
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
                    call x0_func(rotorDiameter(turb), yaw(turb), Ct(turb), alpha, & 
                                & TIturbs(turb), beta, x0)
        
                    ! downstream distance from far wake onset to downstream turbine
                    deltax0 = x - x0
                
                    ! calculate wake spreading parameter at each turbine if desired
                    if (calc_k_star .eqv. .true.) then
                        call k_star_func(TIturbs(turb), k_star(turb))
                        ky_local = k_star(turb)
                        kz_local = k_star(turb)
                    end if

                    ! determine the initial wake angle at the onset of far wake
                    call theta_c_0_func(yaw(turb), Ct(turb), theta_c_0)
        
                    ! horizontal spread
                    call sigmay_func(ky_local, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                    !print *, "sigmay ", sigmay
                    ! vertical spread
                    call sigmaz_func(kz_local, deltax0, rotorDiameter(turb), sigmaz)
                    !print *, "sigmaz ", sigmaz
                    ! horizontal cross-wind wake displacement from hub
                    call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                         & ky_local, kz_local, Ct(turb), sigmay, sigmaz, wake_offset)
                    !print *, "wake_offset ", wake_offset       
                              
                    ! cross wind distance from downstream point location to wake center
                    deltay = LocalRotorPointY*cos(yaw(turbI)) + turbineYw(turbI) - (turbineYw(turb) + wake_offset)
            
                    ! cross wind distance from hub height to height of point of interest
                    deltaz = LocalRotorPointZ + turbineZ(turbI) - turbineZ(turb)
                    !print *, "dx, dy, dz: ", deltax0, deltay
                    !print *, "local y,z : ", LocalRotorPointY, LocalRotorPointZ
                    
                    ! find the final point where the original model is undefined
                    call discontinuity_point_func(x0, rotorDiameter(turb), ky, kz, yaw, Ct, & 
                                                 & discontinuity_point)
                                                 
                    ! far wake region
                    if (x > discontinuity_point) then
    
                        ! velocity difference in the wake
                        call deltav_func(deltay, deltaz, Ct(turb), yaw(turb),  &
                                        & sigmay, sigmaz, rotorDiameter(turb), deltav)  

                    ! near wake region (linearized)
                    else
                        
                        ! determine distance from discontinuity point to far wake onset
                        deltax0_dp = discontinuity_point - x0
                
                        ! horizontal spread at discontinuity point
                        call sigmay_func(ky_local, deltax0_dp, rotorDiameter(turb), yaw(turb), sigmay_dp)
                
                        ! vertical spread at discontinuity point
                        call sigmaz_func(kz_local, deltax0_dp, rotorDiameter(turb), sigmaz_dp)
                
                        ! velocity deficit in the nearwake (linear model)
                        call deltav_near_wake_lin_func(deltay, deltaz, &
                                         & Ct(turb), yaw(turb), sigmay_dp, sigmaz_dp, & 
                                         & rotorDiameter(turb), x, discontinuity_point, sigmay_dp, sigmaz_dp, & 
                                         & deltav)
              
                    end if
                
                    ! combine deficits according to selected method wake combination method
                    call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                               wake_combination_method, deficit_sum)
                
                    if ((deltax0 > 0.0_dp) .and. (TI_calculation_method > 0)) then
                
                        ! calculate TI value at each turbine
                        call added_ti_func(TI, Ct, deltax0, k_star(turb), rotorDiameter(turb), & 
                                           & rotorDiameter(turbI), deltay, turbineZ(turb), &
                                           & turbineZ(turbI), TIturbs(turb), &
                                           & TI_calculation_method, &
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
       
    end do

    !print *, "here 0"
    
    do, loc=1, nSamples
         
        ! set combined velocity deficit to zero for loc
        deficit_sum = 0.0_dp
        
        do, turb=1, nTurbines ! at turbineX-locations
        
            ! set velocity deficit at loc due to turb to zero
            deltav = 0.0_dp

            ! determine the onset location of far wake
            call x0_func(rotorDiameter(turb), yaw(turb), Ct(turb), alpha, TIturbs(turb), &
                        & beta, x0)
            !print *, "x0: ", x0/rotorDiameter(turb)
        
            ! determine the initial wake angle at the onset of far wake
            call theta_c_0_func(yaw(turb), Ct(turb), theta_c_0)
        
            ! downstream distance between turbines
            x = velX(loc) - turbineXw(turb)
                
            ! downstream distance from far wake onset to downstream turbine
            deltax0 = x - x0
            
            if (x > 0.0_dp) then
            
                ! horizontal spread
                call sigmay_func(ky, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
            
                ! vertical spread
                call sigmaz_func(kz, deltax0, rotorDiameter(turb), sigmaz)
            
                ! horizontal cross-wind wake displacement from hub
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky_local(turb), kz_local(turb), Ct(turb), sigmay, &
                                     & sigmaz, wake_offset)
                                 
                ! horizontal distance from downstream hub location to wake center
                deltay = velY(loc) - (turbineYw(turb) + wake_offset)
            
                ! vertical distance from downstream hub location to wake center
                deltaz = velZ(loc) - turbineZ(turb)
                !print *, "here 1"
                
                ! find the final point where the original model is undefined
                call discontinuity_point_func(x0, rotorDiameter(turb), ky, kz, yaw, Ct, & 
                                             & discontinuity_point)
                                             
                ! far wake region
                if (x > discontinuity_point) then

                    ! velocity difference in the wake
                    call deltav_func(deltay, deltaz, Ct(turb), yaw(turb),  &
                                    & sigmay, sigmaz, rotorDiameter(turb), deltav)
                    !print *, "here 2"
                    !print *, 'deficit sum fw: ', deficit_sum
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
                                     & Ct(turb), yaw(turb), sigmay_dp, sigmaz_dp,           & 
                                     & rotorDiameter(turb), x, x0, sigmay_dp, sigmaz_dp,    & 
                                     & deltav)
                                     
                    !print *, "here 3"
                end if
            
                ! combine deficits according to selected method wake combination method
                call wake_combination_func(wind_speed, wtVelocity(turb), deltav,         &
                                           wake_combination_method, deficit_sum) 
                !print *, 'deficit sum nw: ', deficit_sum
            
            end if
            
            ! if (x < 0.0_dp ) then
!                 print *, "deltav: ", deltav
!                 print *, "deficit sum at ", turb, ": ", deficit_sum 
!                 print *, "deltax: ", x
!                 print *, "deltav: ", deltav
!             end if
            
        end do
        
        ! find velocity at point p due to the wake of turbine turb
        point_velocity = wind_speed - deficit_sum
        
        ! put sample point height in global reference frame
        point_z = velZ(loc)
        
        ! adjust sample point velocity for shear
        call wind_shear_func(point_z, point_velocity, z_ref, z_0, shear_exp, point_velocity_with_shear)
        
        ! final velocity calculation for location loc
        wsArray(loc) = point_velocity_with_shear
        
        !print *, 'deficit sum: ', deficit_sum
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
                      & rotor_diameter_ust, deltav) 
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter_ust
     
    ! out
    real(dp), intent(out) :: deltav

    ! load intrinsic functions
    intrinsic cos, sqrt, exp
    
    !print *, "rotor_diameter in deltav entry", rotor_diameter_ust
    
    ! velocity difference in the wake at each sample point
    deltav = (                                                                    &
        (1.0_dp - sqrt(1.0_dp - Ct *                                                         &
                       cos(yaw) / (8.0_dp * sigmay * sigmaz / (rotor_diameter_ust ** 2)))) *     &
        exp(-0.5_dp * (deltay / sigmay) ** 2) * exp(-0.5_dp * (deltaz / sigmaz) ** 2)&
    )
    
    !print *, "rotor_diameter in deltav exit", rotor_diameter_ust

end subroutine deltav_func


! calculates the velocity difference between hub velocity and free stream for a given wake
! for use in the near wake region only
subroutine deltav_near_wake_lin_func(deltay, deltaz, Ct, yaw,  &
                                 & sigmay, sigmaz, rotor_diameter_ust, x, &
                                 & discontinuity_point, sigmay0, sigmaz0, deltav) 
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter_ust
    real(dp), intent(in) :: x, discontinuity_point, sigmay0, sigmaz0
    
    ! local
    real(dp) :: deltav0m, deltavr

    ! out
    real(dp), intent(out) :: deltav

    ! load intrinsic functions
    intrinsic cos, sqrt, exp

    ! magnitude term of gaussian at x0
    deltav0m = (                                         &
                (1.0_dp - sqrt(1.0_dp - Ct *                          &
                cos(yaw) / (8.0_dp * sigmay0 * sigmaz0 /              &
                                            (rotor_diameter_ust ** 2)))))
    ! initialize the gaussian magnitude term at the rotor for the linear interpolation
    deltavr = deltav0m

    ! linearized gaussian magnitude term for near wake
    deltav = (((deltav0m - deltavr)/discontinuity_point) * x + deltavr) *       &
        exp(-0.5_dp * (deltay / sigmay) ** 2) *                                 &
        exp(-0.5_dp * (deltaz / sigmaz) ** 2)
                
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
    real(dp) :: OVdYd, OVr, OVRR, OVL, OVz
    
    ! load intrinsic functions
    intrinsic dacos, sqrt
    
    ! distance between wake center and rotor center
    OVdYd = sqrt((wake_center_y-turbine_y)**2 + (wake_center_z - turbine_z)**2) 

    ! find rotor radius
    OVr = rotor_diameter/2.0_dp
    
    ! find wake radius
    OVRR = wake_diameter/2.0_dp
    
    ! make sure the distance from wake center to turbine hub is positive
    ! OVdYd = abs(OVdYd) !!! commented out since change to 2D distance (y,z) will always be positive
    
    ! calculate the distance from the wake center to the line perpendicular to the 
    ! line between the two circle intersection points
    if (OVdYd >= 0.0_dp + tol) then ! check case to avoid division by zero
        OVL = (-OVr*OVr+OVRR*OVRR+OVdYd*OVdYd)/(2.0_dp*OVdYd)
    else
        OVL = 0.0_dp
    end if

    OVz = OVRR*OVRR-OVL*OVL

    ! Finish calculating the distance from the intersection line to the outer edge of the wake
    if (OVz > 0.0_dp + tol) then
        OVz = sqrt(OVz)
    else
        OVz = 0.0_dp
    end if

    if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake

        if (OVL < OVRR .and. (OVdYd-OVL) < OVr) then
            wake_overlap = OVRR*OVRR*dacos(OVL/OVRR) + OVr*OVr*dacos((OVdYd-OVL)/OVr) - OVdYd*OVz
        else if (OVRR > OVr) then
            wake_overlap = pi*OVr*OVr
        else
            wake_overlap = pi*OVRR*OVRR
        end if
    else
        wake_overlap = 0.0_dp
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
                        & TI_calculation_method, TI_dst)
                                 
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: Ct_ust, x, k_star_ust, rotor_diameter_ust, rotor_diameter_dst
    real(dp), intent(in) :: deltay, wake_height, turbine_height, TI_ust, TI
    integer, intent(in) :: TI_calculation_method
    
    ! local
    real(dp) :: axial_induction_ust, beta, epsilon, sigma, wake_diameter, wake_overlap
    real(dp) :: TI_added, TI_tmp, sum_of_squares, rotor_area_dst
    real(dp), parameter :: pi = 3.141592653589793_dp
    
    ! out  
    real(dp), intent(inout) :: TI_dst
    
    ! intrinsic functions
    intrinsic sqrt
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
        TI_added = 0.73_dp*(axial_induction_ust**0.8325)*(TI_ust**0.0325)* & 
                    ((x/rotor_diameter_ust)**-0.32)
        !print *, "TI_added = ", TI_added
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2
        ! Calculate the total turbulence intensity at the downstream turbine
        !sum_of_squares = TI_dst**2 + (TI_added*wake_overlap)**2
        ! print *, "sum of squares = ", sum_of_squares
!         TI_dst = sqrt(sum_of_squares)
!         print *, "TI_dst = ", TI_dst
        TI_dst = sqrt(TI_dst**2 + (TI_added*wake_overlap/rotor_area_dst)**2)
        
    
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
        TI_added = 0.73_dp*(axial_induction_ust**0.8325)*(TI_ust**0.0325)* & 
                    ((x/rotor_diameter_ust)**-0.32)
        
        ! Calculate the total turbulence intensity at the downstream turbine based on 
        ! current upstream turbine
        rotor_area_dst = 0.25_dp*pi*rotor_diameter_dst**2
        TI_tmp = sqrt(TI**2 + (TI_added*wake_overlap/rotor_area_dst)**2)
        
        ! Check if this is the max and use it if it is
        if (TI_tmp > TI_dst) then
            TI_dst = TI_tmp
        end if
    
    ! TODO add other TI calculation methods
        
    ! wake combination method error 
    else
        print *, "Invalid added TI calculation method. Must be one of [1,2,3]."
        stop 1
    end if                       
    
end subroutine added_ti_func

! compute wake spread parameter based on local turbulence intensity
subroutine k_star_func(TI_ust, k_star_ust)
                                 
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: TI_ust
    
    ! out  
    real(dp), intent(inout) :: k_star_ust
    
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

    ! adjusted wind speed for wind shear
    adjusted_wind_speed = u_ref*((point_z-z_0)/(z_ref-z_0))**shear_exp
    
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

