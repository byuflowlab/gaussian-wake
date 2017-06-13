! implementation of the Bastankhah and Porte Agel (BPA) wake model for analysis
subroutine porteagel_analyze(nTurbines, turbineXw, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, I, wake_combination_method, &
                             TI_calculation_method, wtVelocity)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        Ct yawDeg

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    integer, intent(in) :: wake_combination_method, TI_calculation_method
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, I, wind_speed

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset
    real(dp) :: x, deltav, deltav0m, deltaz, sigmay0, sigmaz0
    Integer :: turb, turbI
    real(dp), parameter :: pi = 3.141592653589793_dp

    ! model out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity

    intrinsic cos, atan, max, sqrt, log

    ! bastankhah and porte agel 2016 define yaw to be positive clockwise, this is reversed
    yaw = - yawDeg*pi/180.0_dp

    ! initialize wind turbine velocity array to the free-stream wind speed
    wtVelocity = wind_speed

    do, turb=1, nTurbines
        
        ! determine the onset location of far wake
        call x0_func(rotorDiameter(turb), yaw(turb), Ct(turb), alpha, I, beta, x0)
        
        ! determine the initial wake angle at the onset of far wake
        call theta_c_0_func(yaw(turb), Ct(turb), theta_c_0)
        
        do, turbI=1, nTurbines ! at turbineX-locations
        
            ! downstream distance between turbines
            x = turbineXw(turbI) - turbineXw(turb)
                
            ! downstream distance from far wake onset to downstream turbine
            deltax0 = x - x0

            ! far wake region
            if (x >= x0) then
            
                ! horizontal spread
                call sigmay_func(ky, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                
                ! vertical spread
                call sigmaz_func(kz, deltax0, rotorDiameter(turb), sigmaz)
                
                ! horizontal cross-wind wake displacement from hub
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky, kz, Ct(turb), sigmay, sigmaz, wake_offset)
                                     
                ! cross wind distance from downstream hub location to wake center
                deltay = turbineYw(turbI) - (turbineYw(turb) + wake_offset)
                
                ! cross wind distance from hub height to height of point of interest
                deltaz = turbineZ(turbI) - turbineZ(turb)
                
                ! velocity difference in the wake
                call deltav_func(deltay, deltaz, wake_offset, wind_speed, Ct(turb), & 
                                 & yaw(turb), sigmay, sigmaz, rotorDiameter(turb), deltav)
                                 
                ! linear wake superposition (additive)
                wtVelocity(turbI) = wtVelocity(turbI) - deltav

            ! near wake region (linearized)
            else if (x > 0.0_dp) then
    
                ! horizontal spread
                call sigmay_func(ky, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                
                ! vertical spread
                call sigmaz_func(kz, deltax0, rotorDiameter(turb), sigmaz)
                                                
                ! horizontal cross-wind wake displacement from hub
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky, kz, Ct(turb), sigmay, sigmaz, wake_offset)
                
                ! distance from downstream hub location to wake center
                deltay = turbineYw(turbI) - (turbineYw(turb) + wake_offset)
                
                ! vertical distance from downstream hub location to wake center
                deltaz = turbineZ(turbI) - turbineZ(turb)
                
                ! horizontal spread at far wake onset
                call sigmay_func(ky, 0.0_dp, rotorDiameter(turb), yaw(turb), sigmay0)
                
                ! vertical spread at far wake onset
                call sigmaz_func(kz, 0.0_dp, rotorDiameter(turb), sigmaz0)
                
                ! velocity deficit in the nearwake (linear model)
                call deltav_near_wake_lin_func(deltay, deltaz, wake_offset, wind_speed, &
                                 & Ct(turb), yaw(turb), sigmay, sigmaz, & 
                                 & rotorDiameter(turb), x, x0, sigmay0, sigmaz0, deltav)
                                 
                ! linear wake superposition (additive)
                wtVelocity(turbI) = wtVelocity(turbI) - deltav

            end if
            
            ! make sure turbine inflow velocity is non-negative
            if (wtVelocity(turbI) .lt. 0.0_dp) then 
                wtVelocity(turbI) = 0.0_dp
            end if
            
        end do
    end do

    !print *, "fortran"

end subroutine porteagel_analyze

! implementation of the Bastankhah and Porte Agel (BPA) wake model for visualization
subroutine porteagel_visualize(nTurbines, nSamples, turbineXw, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, I, velX, velY, velZ, wsArray)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        Ct yawDeg

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines, nSamples
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, I, wind_speed
    real(dp), dimension(nSamples), intent(in) :: velX, velY, velZ

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset
    real(dp) :: x, deltav, deltav0m, sigmay0, sigmaz0, deltavs, deltaz
    Integer :: turb, loc
    real(dp), parameter :: pi = 3.141592653589793_dp

    ! model out
    real(dp), dimension(nSamples), intent(out) :: wsArray

    intrinsic cos, atan, max, sqrt, log

    ! bastankhah and porte agel 2016 define yaw to be positive clockwise, this is reversed
    yaw = - yawDeg*pi/180.0_dp

    ! initialize location velocities to free stream
    wsArray = wind_speed

    do, turb=1, nTurbines
        
         ! determine the onset location of far wake
        call x0_func(rotorDiameter(turb), yaw(turb), Ct(turb), alpha, I, beta, x0)
        
        ! determine the initial wake angle at the onset of far wake
        call theta_c_0_func(yaw(turb), Ct(turb), theta_c_0)
        
        do, loc=1, nSamples ! at turbineX-locations
        
            ! downstream distance between turbines
            x = velX(loc) - turbineXw(turb)
                
            ! downstream distance from far wake onset to downstream turbine
            deltax0 = x - x0

            ! far wake region
            if (x >= x0) then
                
                ! horizontal spread
                call sigmay_func(ky, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                
                ! vertical spread
                call sigmaz_func(kz, deltax0, rotorDiameter(turb), sigmaz)
                
                ! horizontal cross-wind wake displacement from hub
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky, kz, Ct(turb), sigmay, sigmaz, wake_offset)
                                     
                ! horizontal distance from downstream hub location to wake center
                deltay = velY(loc) - (turbineYw(turb) + wake_offset)
                
                ! vertical distance from downstream hub location to wake center
                deltaz = velZ(loc) - turbineZ(turb)
                
                ! velocity difference in the wake
                call deltav_func(deltay, deltaz, wake_offset, wind_speed, Ct(turb), & 
                                 & yaw(turb), sigmay, sigmaz, rotorDiameter(turb), deltav)
                                 
                ! linear wake superposition (additive)
                wsArray(loc) = wsArray(loc) - deltav

            ! near wake region (linearized)
            else if (x > 0.0_dp) then

                ! horizontal spread at point of interest
                call sigmay_func(ky, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                
                ! vertical spread at point of interest
                call sigmaz_func(kz, deltax0, rotorDiameter(turb), sigmaz)
                                                
                ! horizontal cross-wind wake displacement from hub
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky, kz, Ct(turb), sigmay, sigmaz, wake_offset)
                
                ! horizontal distance from downstream hub location to wake center
                deltay = velY(loc) - (turbineYw(turb) + wake_offset)
                
                ! vertical distance from downstream hub location to wake center
                deltaz = velZ(loc) - turbineZ(turb)
                
                ! horizontal spread at far wake onset
                call sigmay_func(ky, 0.0_dp, rotorDiameter(turb), yaw(turb), sigmay0)
                
                ! vertical spread at far wake onset
                call sigmaz_func(kz, 0.0_dp, rotorDiameter(turb), sigmaz0)
                
                ! velocity deficit in the nearwake (linear model)
                call deltav_near_wake_lin_func(deltay, deltaz, wake_offset, wind_speed, &
                                 & Ct(turb), yaw(turb), sigmay, sigmaz, & 
                                 & rotorDiameter(turb), x, x0, sigmay0, sigmaz0, deltav)
                                
                ! linear wake superposition (additive)
                wsArray(loc) = wsArray(loc) - deltav

            end if
            
            ! make sure turbine flow-field velocity is non-negative
            if (wsArray(loc) .lt. 0.0_dp) then 
                wsArray(loc) = 0.0_dp
            end if
            
        end do
    end do

    !print *, "fortran"

end subroutine porteagel_visualize


! calculates the onset of far-wake conditions
subroutine x0_func(rotor_diameter, yaw, Ct, alpha, I, beta, x0)

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: rotor_diameter, yaw, Ct, alpha, I, beta

    ! out
    real(dp), intent(out) :: x0

    intrinsic cos, sqrt, log
                            

    ! determine the onset location of far wake
    x0 = rotor_diameter * (cos(yaw) * (1.0_dp + sqrt(1.0_dp - Ct)) / &
                                (sqrt(2.0_dp) * (alpha * I + beta * &
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
    intrinsic cos, sqrt
    
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
subroutine deltav_func(deltay, deltaz, wake_offset, wind_speed, Ct, yaw, sigmay, sigmaz, &
                       & rotor_diameter, deltav) 
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, wake_offset, wind_speed, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter
    
    ! out
    real(dp), intent(out) :: deltav

    intrinsic cos, sqrt, exp
    
    ! velocity difference in the wake
    deltav = wind_speed * (                                                 &
        (1.0_dp - sqrt(1.0_dp - Ct *                                                     &
                       cos(yaw) / (8.0_dp * sigmay * sigmaz / (rotor_diameter ** 2)))) * &
        exp(-0.5_dp * ((deltay) / sigmay) ** 2) * exp(-0.5_dp * ((deltaz) / sigmaz) ** 2)&
    )
    
end subroutine deltav_func


! calculates the velocity difference between hub velocity and free stream for a given wake
! for use in the near wake region only
subroutine deltav_near_wake_lin_func(deltay, deltaz, wake_offset, wind_speed, Ct, yaw, &
                                 & sigmay, sigmaz, rotor_diameter, x, x0, &
                                 & sigmay0, sigmaz0, deltav) 
                       
    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    real(dp), intent(in) :: deltay, deltaz, wake_offset, wind_speed, Ct, yaw, sigmay
    real(dp), intent(in) :: sigmaz, rotor_diameter, x, x0, sigmay0, sigmaz0
    
    ! local
    real(dp) :: deltav0m, deltavs
    
    ! out
    real(dp), intent(out) :: deltav

    intrinsic cos, sqrt, exp
    
    deltavs = 0.9*wind_speed

    ! magnitude term of gaussian at x0
    deltav0m = wind_speed * (                                         &
                (1.0_dp - sqrt(1.0_dp - Ct *                          &
                cos(yaw) / (8.0_dp * sigmay0 * sigmaz0 /              &
                                            (rotor_diameter ** 2)))))
                                            
    ! linearized gaussian magnitude term for near wake
    deltav = (((deltav0m - deltavs)/x0) * x + deltavs) *              &
        exp(-0.5_dp * (deltay / sigmay) ** 2) *                       &
        exp(-0.5_dp * (deltaz / sigmaz) ** 2)
    ! deltav = deltav0m *              &
!         exp(-0.5_dp * (deltay / sigmay) ** 2) *                       &
!         exp(-0.5_dp * (deltaz / sigmaz) ** 2)
    
end subroutine deltav_near_wake_lin_func

! calculates the overlap area between a given wake and a rotor area
subroutine overlap_area_func(turbine_y, turbine_z, rotor_diameter, &
                            wake_center_y, wake_center_z, wake_diameter, &
                            wake_overlap)
!   calculate overlap of rotors and wake zones (wake zone location defined by wake 
!   center and wake diameter)
!   turbineX,turbineY is x,y-location of center of rotor

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
    
    intrinsic dacos
    
    OVdYd = wake_center_y-turbine_y        ! distance between wake center and rotor center
    OVr = rotor_diameter/2.0_dp            ! rotor diameter
    OVRR = wake_diameter/2.0_dp            ! wake diameter
    OVdYd = abs(OVdYd)
    if (OVdYd >= 0.0_dp + tol) then
        ! calculate the distance from the wake center to the vertical line between
        ! the two circle intersection points
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

    if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake zone

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

! combine wakes using various methods
subroutine wake_combination_func(Uinf, Ueffu, Ueffd, tmp, wake_combination_method, &
                                 deficit)
    ! combines wakes to calculate velocity at a given turbine
    ! Uinf      = Free stream velocity
    ! Ueffu     = Effective velocity as seen by the upstream rotor
    ! Ueffd     = Current effective velocity as seen by the downstream rotor
    ! tmp       =
    ! velocity  = new effective velocity at this turbine
     
    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    real(dp), intent(in) :: Uinf, Ueffu, Ueffd, tmp
    integer, intent(in) :: wake_combination_method
    
    ! out    
    real(dp), intent(out) :: deficit
    
    ! local
    real(dp), parameter :: pi = 3.141592653589793_dp
    
    intrinsic sqrt
    
    ! freestream linear superposition
    if (wake_combination_method == 0) then
        deficit = (Uinf - Ueffd) + tmp

    ! local velocity linear superposition
    else if (wake_combination_method == 1) then
        deficit = (Ueffu - Ueffd) + tmp
        
    ! sum of squares freestream superposition
    else if (wake_combination_method == 2) then 
        deficit = sqrt((Uinf - Ueffd)**2 + tmp**2)
    
    !sum of squares local velocity superposition
    else if (wake_combination_method == 3) then
        deficit = sqrt((Ueffu - Ueffd)**2 + tmp**2)
    
    ! error
    else
        print *, "Invalid wake combination method. Must be one of [0,1,2,3]."
        stop 1
    end if                       
    
end subroutine wake_combination_func
