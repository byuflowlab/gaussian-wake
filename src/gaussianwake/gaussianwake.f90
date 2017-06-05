subroutine porteagel_analyze(nTurbines, turbineXw, turbineYw, turbineZ, &
                             rotorDiameter, Ct, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, I, wtVelocity)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        Ct yawDeg

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct
    real(dp), intent(in) :: ky, kz, alpha, beta, I, wind_speed

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset
    real(dp) :: x, deltav, deltav0m
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
        x0 = rotorDiameter(turb) * (cos(yaw(turb)) * (1.0_dp + sqrt(1.0_dp - Ct(turb))) / &
                                    (sqrt(2.0_dp) * (alpha * I + beta * (1.0_dp - sqrt(1.0_dp - Ct(turb))))))
        
        ! determine the initial wake angle at the onset of far wake
        theta_c_0 = 0.3_dp * yaw(turb) * (1.0_dp - sqrt(1.0_dp - Ct(turb) * cos(yaw(turb)))) / cos(yaw(turb))

        do, turbI=1, nTurbines ! at turbineX-locations
        
            ! downstream distance between turbines
            x = turbineXw(turbI) - turbineXw(turb)
                
            ! downstream distance from far wake onset to downstream turbine
            deltax0 = x - x0


            ! far wake region
            if (deltax0 > 0.0_dp) then
            
                ! horizontal spread
                ! sigmay = rotorDiameter(turb) * (ky * deltax0 / rotorDiameter(turb) &
!                                                 + cos(yaw(turb)) / sqrt(8.0_dp))
                call sigmay_func(ky, deltax0, rotorDiameter(turb), yaw(turb), sigmay)
                
                ! vertical spread
                ! sigmaz = rotorDiameter(turb) * (kz * deltax0 / rotorDiameter(turb) &
!                                                 + 1.0_dp / sqrt(8.0_dp))
                call sigmaz_func(kz, deltax0, rotorDiameter(turb), sigmaz)
                
                ! horizontal cross-wind wake displacement from hub
                ! wake_offset = rotorDiameter(turb) * (                           &
!                     theta_c_0 * x0 / rotorDiameter(turb) +                      &
!                     (theta_c_0 / 14.7_dp) * sqrt(cos(yaw(turb)) / (ky * kz * Ct(turb))) * &
!                     (2.9_dp + 1.3_dp * sqrt(1.0_dp - Ct(turb)) - Ct(turb)) *    &
!                     log(                                                        &
!                         ((1.6_dp + sqrt(Ct(turb))) *                            &
!                          (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
!                                         (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
!                           - sqrt(Ct(turb)))) /                                  &
!                         ((1.6_dp - sqrt(Ct(turb))) *                            &
!                          (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
!                                         (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
!                           + sqrt(Ct(turb))))                                    &
!                     )                                                           &
!                 )
                call wake_offset_func(rotorDiameter(turb), theta_c_0, x0, yaw(turb), &
                                     & ky, kz, Ct(turb), sigmay, sigmaz, wake_offset)
                                     
                ! distance from downstream hub location to wake center
                deltay = turbineYw(turbI) - (turbineYw(turb) + wake_offset)
                ! velocity difference in the wake
                deltav = wind_speed * (                                         &
                    (1.0_dp - sqrt(1.0_dp - Ct(turb) *                          &
                                   cos(yaw(turb)) / (8.0_dp * sigmay * sigmaz / &
                                                        (rotorDiameter(turb) ** 2)))) *  &
                    exp(-0.5_dp * ((deltay) / sigmay) ** 2) *                   &
                    exp(-0.5_dp * ((turbineZ(turbI) - turbineZ(turb)) / sigmaz) ** 2) &
                )
                ! linear wake superposition (additive)
                wtVelocity(turbI) = wtVelocity(turbI) - deltav

            ! near wake region (linearized)
            else if (deltax0 > -x0) then

                ! horizontal spread
                sigmay = rotorDiameter(turb) * (ky * deltax0 / rotorDiameter(turb) &
                                                + cos(yaw(turb)) / sqrt(8.0_dp))
                ! vertical spread
                sigmaz = rotorDiameter(turb) * (kz * deltax0 / rotorDiameter(turb) &
                                                + 1.0_dp / sqrt(8.0_dp))
                                                
                ! horizontal cross-wind wake displacement from hub
                wake_offset = rotorDiameter(turb) * (                           &
                    theta_c_0 * x0 / rotorDiameter(turb) +                      &
                    (theta_c_0 / 14.7_dp) * sqrt(cos(yaw(turb)) / (ky * kz * Ct(turb))) * &
                    (2.9_dp + 1.3_dp * sqrt(1.0_dp - Ct(turb)) - Ct(turb)) *    &
                    log(                                                        &
                        ((1.6_dp + sqrt(Ct(turb))) *                            &
                         (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
                                        (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
                          - sqrt(Ct(turb)))) /                                  &
                        ((1.6_dp - sqrt(Ct(turb))) *                            &
                         (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
                                        (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
                          + sqrt(Ct(turb))))                                    &
                    )                                                           &
                )
                
                ! distance from downstream hub location to wake center
                deltay = turbineYw(turbI) - (turbineYw(turb) + wake_offset)
                
                ! magnitude term of gaussian at x0
                deltav0m = wind_speed * (                                         &
                           (1.0_dp - sqrt(1.0_dp - Ct(turb) *                     &
                           cos(yaw(turb)) / (8.0_dp * sigmay * sigmaz /           &
                                                        (rotorDiameter(turb) ** 2)))))
                                                        
                ! linearized gaussian magnitude term for near wake
                deltav = (deltav0m/x0) * x *  &
                    exp(-0.5_dp * ((deltay) / sigmay) ** 2) *                   &
                    exp(-0.5_dp * ((turbineZ(turbI) - turbineZ(turb)) / sigmaz) ** 2)
                
                ! linear wake superposition (additive)
                wtVelocity(turbI) = wtVelocity(turbI) - deltav

                ! first try
                ! xpc = deltax0 + x0
! 
!                 ! wind speed at onset of far wake
!                 u_0 = wind_speed * sqrt(1.0_dp - Ct(turb))
! 
!                 ! velocity deficit at the wake center at onset of far wake
!                 C_0 = 1.0_dp - u_0 / wind_speed
! 
!                 ! wind velocity at the rotor
!                 u_r = Ct(turb)*cos(yaw(turb))/(2.0_dp*(1.0_dp - sqrt(1.0_dp - Ct(turb)*cos(yaw(turb)))))
! 
!                 ! potential core heighth at the rotor
!                 z_r = rotorDiameter(turb)*sqrt(u_r/wind_speed)
! 
!                 ! potential core width at the rotor
!                 y_r = rotorDiameter(turb)*cos(yaw(turb))*sqrt(u_r/wind_speed)
! 
!                 sigmaz_0 = 0.5_dp*rotorDiameter(turb)*sqrt(u_r/(wind_speed+u_0)
!                 sigmay_0 = rotorDiameter(turb) * (ky * x0 / rotorDiameter(turb) &
!                                                 + cos(yaw(turb)) / sqrt(8.0_dp))
!                 s = (sigmay_0/x0)*(deltax0+x0)
!                 r_pc =

            end if
        end do
    end do

    !print *, "fortran"

end subroutine porteagel_analyze


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
    real(dp) :: x, deltav, deltav0m, sigmay0, sigmaz0, deltavs
    Integer :: turb, loc, t
    real(dp), parameter :: pi = 3.141592653589793_dp

    ! model out
    real(dp), dimension(nSamples), intent(out) :: wsArray

    intrinsic cos, atan, max, sqrt, log

    ! bastankhah and porte agel 2016 define yaw to be positive clockwise, this is reversed
    yaw = - yawDeg*pi/180.0_dp

    ! initialize location velocities to free stream
    wsArray = wind_speed

    do, turb=1, nTurbines
        
        ! downstream distance to far wake onset
        x0 = rotorDiameter(turb) * (cos(yaw(turb)) * (1.0_dp + sqrt(1.0_dp - Ct(turb))) / &
                                    (sqrt(2.0_dp) * (alpha * I + beta * (1.0_dp - sqrt(1.0_dp - Ct(turb))))))
        
        ! initial wake angle at far wake onset
        theta_c_0 = 0.3_dp * yaw(turb) * (1.0_dp - sqrt(1.0_dp - Ct(turb) * cos(yaw(turb)))) / cos(yaw(turb))
        
        ! horizontal spread at far wake onset
        sigmay0 = rotorDiameter(turb) * (ky * 0.0_dp / rotorDiameter(turb) &
                                        + cos(yaw(turb)) / sqrt(8.0_dp))
        ! vertical spread
        sigmaz0 = rotorDiameter(turb) * (kz * 0.0_dp / rotorDiameter(turb) &
                                        + 1.0_dp / sqrt(8.0_dp))
                                        
        ! magnitude term of gaussian at x0
        deltav0m = wind_speed * (                                         &
                    (1.0_dp - sqrt(1.0_dp - Ct(turb) *                     &
                    cos(yaw(turb)) / (8.0_dp * sigmay0 * sigmaz0 /           &
                                                (rotorDiameter(turb) ** 2)))))

        deltavs = 0.9*wind_speed
        
        do, loc=1, nSamples ! at turbineX-locations
            t = 0
            ! downstream distance between turbines
            x = velX(loc) - turbineXw(turb)
                
            ! downstream distance from far wake onset to downstream turbine
            deltax0 = x - x0
            
!              print *, x0, x, deltax0

            ! far wake region
            if (x >= x0) then
                t = t + 1
!                 print *, "here, here"
                ! horizontal spread
                sigmay = rotorDiameter(turb) * (ky * deltax0 / rotorDiameter(turb) &
                                                + cos(yaw(turb)) / sqrt(8.0_dp))
                ! vertical spread
                sigmaz = rotorDiameter(turb) * (kz * deltax0 / rotorDiameter(turb) &
                                                + 1.0_dp / sqrt(8.0_dp))
                ! horizontal cross-wind wake displacement from hub
                wake_offset = rotorDiameter(turb) * (                           &
                    theta_c_0 * x0 / rotorDiameter(turb) +                      &
                    (theta_c_0 / 14.7_dp) * sqrt(cos(yaw(turb)) / (ky * kz * Ct(turb))) * &
                    (2.9_dp + 1.3_dp * sqrt(1.0_dp - Ct(turb)) - Ct(turb)) *    &
                    log(                                                        &
                        ((1.6_dp + sqrt(Ct(turb))) *                            &
                         (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
                                        (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
                          - sqrt(Ct(turb)))) /                                  &
                        ((1.6_dp - sqrt(Ct(turb))) *                            &
                         (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
                                        (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
                          + sqrt(Ct(turb))))                                    &
                    )                                                           &
                )
                ! distance from downstream hub location to wake center
                deltay = velY(loc) - (turbineYw(turb) + wake_offset)
                ! velocity difference in the wake
                deltav = wind_speed * (                                         &
                    (1.0_dp - sqrt(1.0_dp - Ct(turb) *                          &
                                   cos(yaw(turb)) / (8.0_dp * sigmay * sigmaz / &
                                                        (rotorDiameter(turb) ** 2)))) *  &
                    exp(-0.5_dp * ((deltay) / sigmay) ** 2) *                   &
                    exp(-0.5_dp * ((velZ(loc) - turbineZ(turb)) / sigmaz) ** 2) &
                )
                ! linear wake superposition (additive)
                wsArray(loc) = wsArray(loc) - deltav

            ! near wake region (linearized)
            else if (x > 0.0_dp) then
!             else if (x > x0) then
                t = t + 1
                ! horizontal spread
                sigmay = rotorDiameter(turb) * (ky * deltax0 / rotorDiameter(turb) &
                                                + cos(yaw(turb)) / sqrt(8.0_dp))
                ! vertical spread
                sigmaz = rotorDiameter(turb) * (kz * deltax0 / rotorDiameter(turb) &
                                                + 1.0_dp / sqrt(8.0_dp))
                                                
                ! horizontal cross-wind wake displacement from hub
                wake_offset = rotorDiameter(turb) * (                           &
                    (theta_c_0 * x0 / rotorDiameter(turb)) +                      &
                    (theta_c_0 / 14.7_dp) * sqrt(cos(yaw(turb)) / (ky * kz * Ct(turb))) * &
                    (2.9_dp + 1.3_dp * sqrt(1.0_dp - Ct(turb)) - Ct(turb)) *    &
                    log(                                                        &
                        ((1.6_dp + sqrt(Ct(turb))) *                            &
                         (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
                                        (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
                          - sqrt(Ct(turb)))) /                                  &
                        ((1.6_dp - sqrt(Ct(turb))) *                            &
                         (1.6_dp * sqrt(8.0_dp * sigmay * sigmaz /              &
                                        (cos(yaw(turb)) * rotorDiameter(turb) ** 2)) &
                          + sqrt(Ct(turb))))                                    &
                    )                                                           &
                )
                
                ! distance from downstream hub location to wake center
                deltay = velY(loc) - (turbineYw(turb) + wake_offset)
                                                        
                ! linearized gaussian magnitude term for near wake
                deltav = (((deltav0m - deltavs)/x0) * x + deltavs) *  &
                    exp(-0.5_dp * ((deltay) / sigmay) ** 2) *                   &
                    exp(-0.5_dp * ((velZ(loc) - turbineZ(turb)) / sigmaz) ** 2)
                print *, deltav, deltav0m, x/rotorDiameter(turb)
                ! linear wake superposition (additive)
                wsArray(loc) = wsArray(loc) - deltav

                ! first try
                ! xpc = deltax0 + x0
! 
!                 ! wind speed at onset of far wake
!                 u_0 = wind_speed * sqrt(1.0_dp - Ct(turb))
! 
!                 ! velocity deficit at the wake center at onset of far wake
!                 C_0 = 1.0_dp - u_0 / wind_speed
! 
!                 ! wind velocity at the rotor
!                 u_r = Ct(turb)*cos(yaw(turb))/(2.0_dp*(1.0_dp - sqrt(1.0_dp - Ct(turb)*cos(yaw(turb)))))
! 
!                 ! potential core heighth at the rotor
!                 z_r = rotorDiameter(turb)*sqrt(u_r/wind_speed)
! 
!                 ! potential core width at the rotor
!                 y_r = rotorDiameter(turb)*cos(yaw(turb))*sqrt(u_r/wind_speed)
! 
!                 sigmaz_0 = 0.5_dp*rotorDiameter(turb)*sqrt(u_r/(wind_speed+u_0)
!                 sigmay_0 = rotorDiameter(turb) * (ky * x0 / rotorDiameter(turb) &
!                                                 + cos(yaw(turb)) / sqrt(8.0_dp))
!                 s = (sigmay_0/x0)*(deltax0+x0)
!                 r_pc =

            end if
            
            if (t .eq. 2) then
                print *, t, velX(loc), velY(loc), x0/rotorDiameter(turb)
            end if
        end do
    end do

    !print *, "fortran"

end subroutine porteagel_visualize

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
    deltav = wind_speed * (                                                              &
        (1.0_dp - sqrt(1.0_dp - Ct *                                                     &
                       cos(yaw) / (8.0_dp * sigmay * sigmaz / (rotor_diameter ** 2)))) * &
        exp(-0.5_dp * ((deltay) / sigmay) ** 2) * exp(-0.5_dp * ((deltaz) / sigmaz) ** 2)&
    )

end subroutine deltav_func

!        Generated by TAPENADE     (INRIA, Ecuador team)
!  Tapenade 3.12 (r6213) - 13 Oct 2016 10:54
!
!  Differentiation of porteagel_analyze in reverse (adjoint) mode:
!   gradient     of useful results: wtvelocity
!   with respect to varying inputs: rotordiameter turbinez turbinexw
!                wtvelocity turbineyw yawdeg ct
!   RW status of diff variables: rotordiameter:out turbinez:out
!                turbinexw:out wtvelocity:in-zero turbineyw:out
!                yawdeg:out ct:out
SUBROUTINE PORTEAGEL_ANALYZE_BV(nturbines, turbinexw, turbinexwb, &
& turbineyw, turbineywb, turbinez, turbinezb, rotordiameter, &
& rotordiameterb, ct, ctb, wind_speed, yawdeg, yawdegb, ky, kz, alpha, &
& beta, i, wtvelocity, wtvelocityb, nbdirs)
!  Hint: nbdirs should be the maximum number of differentiation directions
  IMPLICIT NONE
!print *, "fortran"
! define precision to be the standard for a double precision ! on local system
  INTEGER, PARAMETER :: dp=KIND(0.d0)
! in
  INTEGER, INTENT(IN) :: nturbines
  REAL(dp), DIMENSION(nturbines), INTENT(IN) :: turbinexw, turbineyw, &
& turbinez
  REAL(dp), DIMENSION(nbdirs, nturbines), intent(out) :: turbinexwb, turbineywb, &
& turbinezb
  REAL(dp), DIMENSION(nturbines), INTENT(IN) :: rotordiameter, yawdeg
  REAL(dp), DIMENSION(nbdirs, nturbines), intent(out) :: rotordiameterb, yawdegb
  REAL(dp), DIMENSION(nturbines), INTENT(IN) :: ct
  REAL(dp), DIMENSION(nbdirs, nturbines), intent(out) :: ctb
  REAL(dp), INTENT(IN) :: ky, kz, alpha, beta, i, wind_speed
! local (General)
  REAL(dp), DIMENSION(nturbines) :: yaw
  REAL(dp), DIMENSION(nbdirs, nturbines) :: yawb
  REAL(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, &
& wake_offset
  REAL(dp), DIMENSION(nbdirs) :: x0b, deltax0b, deltayb, theta_c_0b, &
& sigmayb, sigmazb, wake_offsetb
  REAL(dp) :: deltav
  REAL(dp), DIMENSION(nbdirs) :: deltavb
  INTEGER :: turb, turbi
  REAL(dp), PARAMETER :: pi=3.141592653589793_dp
! model out
  REAL(dp), DIMENSION(nturbines), intent(out) :: wtvelocity
  REAL(dp), DIMENSION(nbdirs, nturbines) :: wtvelocityb
  INTRINSIC COS, ATAN, MAX, SQRT, LOG
  INTRINSIC KIND
  INTRINSIC EXP
  INTEGER :: nd
  REAL(dp) :: temp
  REAL(dp) :: temp0
  REAL(dp) :: temp1
  REAL(dp) :: temp2
  REAL(dp) :: temp3
  REAL(dp) :: temp4
  REAL(dp) :: temp5
  REAL(dp) :: temp6
  REAL(dp) :: temp7
  REAL(dp) :: temp8
  REAL(dp), DIMENSION(nbdirs) :: tempb
  REAL(dp), DIMENSION(nbdirs) :: tempb0
  REAL(dp), DIMENSION(nbdirs) :: tempb1
  REAL(dp) :: temp9
  REAL(dp) :: temp10
  REAL(dp) :: temp11
  REAL(dp) :: temp12
  REAL(dp) :: temp13
  REAL(dp) :: temp14
  REAL(dp) :: temp15
  REAL(dp) :: temp16
  REAL(dp) :: temp17
  REAL(dp) :: temp18
  REAL(dp) :: temp19
  REAL(dp) :: temp20
  REAL(dp) :: temp21
  REAL(dp) :: temp22
  REAL(dp) :: temp23
  REAL(dp) :: temp24
  REAL(dp) :: temp25
  REAL(dp) :: temp26
  REAL(dp) :: temp27
  REAL(dp) :: temp28
  REAL(dp) :: temp29
  REAL(dp) :: temp30
  REAL(dp) :: temp31
  REAL(dp) :: temp32
  REAL(dp) :: temp33
  REAL(dp) :: temp34
  REAL(dp) :: temp35
  REAL(dp) :: temp36
  REAL(dp) :: temp37
  REAL(dp) :: temp38
  REAL(dp) :: temp39
  REAL(dp) :: temp40
  REAL(dp) :: temp41
  REAL(dp) :: temp42
  REAL(dp) :: temp43
  REAL(dp) :: temp44
  REAL(dp) :: temp45
  REAL(dp) :: temp46
  REAL(dp) :: temp47
  REAL(dp), DIMENSION(nbdirs) :: tempb2
  REAL(dp), DIMENSION(nbdirs) :: tempb3
  REAL(dp), DIMENSION(nbdirs) :: tempb4
  REAL(dp), DIMENSION(nbdirs) :: tempb5
  REAL(dp), DIMENSION(nbdirs) :: tempb6
  REAL(dp), DIMENSION(nbdirs) :: tempb7
  REAL(dp), DIMENSION(nbdirs) :: tempb8
  REAL(dp), DIMENSION(nbdirs) :: tempb9
  REAL(dp), DIMENSION(nbdirs) :: tempb10
  REAL(dp), DIMENSION(nbdirs) :: tempb11
  REAL(dp), DIMENSION(nbdirs) :: tempb12
  REAL(dp), DIMENSION(nbdirs) :: tempb13
  REAL(dp), DIMENSION(nbdirs) :: tempb14
  REAL(dp), DIMENSION(nbdirs) :: tempb15
  REAL(dp), DIMENSION(nbdirs) :: tempb16
  REAL(dp), DIMENSION(nbdirs) :: tempb17
  REAL(dp), DIMENSION(nbdirs) :: tempb18
  REAL(dp), DIMENSION(nbdirs) :: tempb19
  REAL(dp), DIMENSION(nbdirs) :: tempb20
  REAL(dp), DIMENSION(nbdirs) :: tempb21
  INTEGER :: branch
  INTEGER :: nbdirs
  yaw = -(yawdeg*pi/180.0_dp)
  DO turb=1,nturbines
    CALL PUSHREAL4ARRAY(x0, dp/4)
    x0 = rotordiameter(turb)*(COS(yaw(turb))*(1.0_dp+SQRT(1.0_dp-ct(turb&
&     )))/(SQRT(2.0_dp)*(alpha*i+beta*(1.0_dp-SQRT(1.0_dp-ct(turb))))))
    CALL PUSHREAL4ARRAY(theta_c_0, dp/4)
    theta_c_0 = 0.3_dp*yaw(turb)*(1.0_dp-SQRT(1.0_dp-ct(turb)*COS(yaw(&
&     turb))))/COS(yaw(turb))
! at turbineX-locations
    DO turbi=1,nturbines
      CALL PUSHREAL4ARRAY(deltax0, dp/4)
      deltax0 = turbinexw(turbi) - (turbinexw(turb)+x0)
      IF (deltax0 .GT. 0.0_dp) THEN
        CALL PUSHREAL4ARRAY(sigmay, dp/4)
        sigmay = rotordiameter(turb)*(ky*deltax0/rotordiameter(turb)+COS&
&         (yaw(turb))/SQRT(8.0_dp))
        CALL PUSHREAL4ARRAY(sigmaz, dp/4)
        sigmaz = rotordiameter(turb)*(kz*deltax0/rotordiameter(turb)+&
&         1.0_dp/SQRT(8.0_dp))
        wake_offset = rotordiameter(turb)*(theta_c_0*x0/rotordiameter(&
&         turb)+theta_c_0/14.7_dp*SQRT(COS(yaw(turb))/(ky*kz*ct(turb)))*&
&         (2.9_dp+1.3_dp*SQRT(1.0_dp-ct(turb))-ct(turb))*LOG((1.6_dp+&
&         SQRT(ct(turb)))*(1.6_dp*SQRT(8.0_dp*sigmay*sigmaz/(COS(yaw(&
&         turb))*rotordiameter(turb)**2))-SQRT(ct(turb)))/((1.6_dp-SQRT(&
&         ct(turb)))*(1.6_dp*SQRT(8.0_dp*sigmay*sigmaz/(COS(yaw(turb))*&
&         rotordiameter(turb)**2))+SQRT(ct(turb))))))
        CALL PUSHREAL4ARRAY(deltay, dp/4)
        deltay = turbineyw(turbi) - (turbineyw(turb)+wake_offset)
        CALL PUSHCONTROL1B(1)
      ELSE
        CALL PUSHCONTROL1B(0)
      END IF
    END DO
  END DO
  DO nd=1,nbdirs
    rotordiameterb(nd, :) = 0.0
    turbinezb(nd, :) = 0.0
    turbinexwb(nd, :) = 0.0
    turbineywb(nd, :) = 0.0
    ctb(nd, :) = 0.0
    yawb(nd, :) = 0.0
  END DO
  DO turb=nturbines,1,-1
    DO nd=1,nbdirs
      theta_c_0b(nd) = 0.0
      x0b(nd) = 0.0
    END DO
    DO turbi=nturbines,1,-1
      CALL POPCONTROL1B(branch)
      IF (branch .EQ. 0) THEN
        DO nd=1,nbdirs
          deltax0b(nd) = 0.0
        END DO
      ELSE
        temp47 = sigmaz**2
        temp40 = turbinez(turbi) - turbinez(turb)
        temp39 = temp40**2/temp47
        temp46 = EXP(-(0.5_dp*temp39))
        temp45 = sigmay**2
        temp38 = deltay**2/temp45
        temp44 = EXP(-(0.5_dp*temp38))
        temp37 = 8.0_dp*sigmay*sigmaz
        temp43 = ct(turb)*rotordiameter(turb)**2
        temp36 = temp43/temp37
        temp42 = COS(yaw(turb))
        temp41 = SQRT(-(temp42*temp36) + 1.0_dp)
        temp35 = SQRT(ct(turb))
        temp34 = COS(yaw(turb))
        temp20 = temp34*rotordiameter(turb)**2
        temp19 = sigmay*sigmaz/temp20
        temp33 = SQRT(8.0_dp*temp19)
        temp32 = SQRT(ct(turb))
        temp18 = (-temp32+1.6_dp)*(1.6_dp*temp33+temp35)
        temp31 = SQRT(ct(turb))
        temp30 = COS(yaw(turb))
        temp17 = temp30*rotordiameter(turb)**2
        temp16 = sigmay*sigmaz/temp17
        temp29 = SQRT(8.0_dp*temp16)
        temp28 = 1.6_dp*temp29 - temp31
        temp27 = SQRT(ct(turb))
        temp15 = (temp27+1.6_dp)*temp28/temp18
        temp26 = LOG(temp15)
        temp25 = SQRT(-ct(turb) + 1.0_dp)
        temp14 = 1.3_dp*temp25 - ct(turb) + 2.9_dp
        temp24 = theta_c_0*temp14/14.7_dp
        temp23 = ky*kz*ct(turb)
        temp22 = COS(yaw(turb))
        temp13 = temp22/temp23
        temp21 = SQRT(temp13)
        temp12 = theta_c_0*x0/rotordiameter(turb)
        deltax0 = turbinexw(turbi) - (turbinexw(turb)+x0)
        temp11 = deltax0/rotordiameter(turb)
        temp10 = SQRT(8.0_dp)
        temp9 = deltax0/rotordiameter(turb)
        DO nd=1,nbdirs
          deltavb(nd) = -wtvelocityb(nd, turbi)
          IF (1.0_dp - temp42*temp36 .EQ. 0.0) THEN
            tempb2(nd) = 0.0
          ELSE
            tempb2(nd) = -(temp44*temp46*wind_speed*deltavb(nd)/(2.0*&
&             temp41))
          END IF
          tempb3(nd) = -(temp42*tempb2(nd)/temp37)
          tempb4(nd) = -(temp36*tempb3(nd))
          tempb5(nd) = wind_speed*(1.0_dp-temp41)*deltavb(nd)
          tempb6(nd) = -(0.5_dp*EXP(-(0.5_dp*temp38))*temp46*tempb5(nd)/&
&           temp45)
          tempb7(nd) = -(0.5_dp*EXP(-(0.5_dp*temp39))*temp44*tempb5(nd)/&
&           temp47)
          tempb8(nd) = 2*temp40*tempb7(nd)
          ctb(nd, turb) = ctb(nd, turb) + rotordiameter(turb)**2*tempb3(&
&           nd)
          deltayb(nd) = 2*deltay*tempb6(nd)
          turbinezb(nd, turbi) = turbinezb(nd, turbi) + tempb8(nd)
          turbinezb(nd, turb) = turbinezb(nd, turb) - tempb8(nd)
          turbineywb(nd, turbi) = turbineywb(nd, turbi) + deltayb(nd)
          turbineywb(nd, turb) = turbineywb(nd, turb) - deltayb(nd)
          wake_offsetb(nd) = -deltayb(nd)
          tempb17(nd) = rotordiameter(turb)*wake_offsetb(nd)
          tempb12(nd) = tempb17(nd)/rotordiameter(turb)
          tempb18(nd) = temp26*tempb17(nd)
          IF (temp13 .EQ. 0.0) THEN
            tempb9(nd) = 0.0
          ELSE
            tempb9(nd) = temp24*tempb18(nd)/(2.0*temp21*temp23)
          END IF
          tempb19(nd) = temp21*theta_c_0*tempb18(nd)/14.7_dp
          tempb20(nd) = temp21*temp24*tempb17(nd)/(temp15*temp18)
          IF (8.0_dp*temp16 .EQ. 0.0) THEN
            tempb15(nd) = 0.0
          ELSE
            tempb15(nd) = 8.0_dp*1.6_dp*(temp27+1.6_dp)*tempb20(nd)/(2.0&
&             *temp29*temp17)
          END IF
          tempb10(nd) = -(temp16*tempb15(nd))
          tempb21(nd) = -(temp15*tempb20(nd))
          IF (8.0_dp*temp19 .EQ. 0.0) THEN
            tempb16(nd) = 0.0
          ELSE
            tempb16(nd) = 8.0_dp*1.6_dp*(1.6_dp-temp32)*tempb21(nd)/(2.0&
&             *temp33*temp20)
          END IF
          sigmayb(nd) = sigmaz*tempb15(nd) + sigmaz*tempb16(nd) - temp38&
&           *2*sigmay*tempb6(nd) + sigmaz*8.0_dp*tempb4(nd)
          sigmazb(nd) = sigmay*tempb15(nd) + sigmay*tempb16(nd) - temp39&
&           *2*sigmaz*tempb7(nd) + 8.0_dp*sigmay*tempb4(nd)
          tempb11(nd) = -(temp19*tempb16(nd))
          yawb(nd, turb) = yawb(nd, turb) + temp36*SIN(yaw(turb))*tempb2&
&           (nd) - rotordiameter(turb)**2*SIN(yaw(turb))*tempb10(nd) - &
&           rotordiameter(turb)**2*SIN(yaw(turb))*tempb11(nd) - &
&           rotordiameter(turb)*SIN(yaw(turb))*sigmayb(nd)/temp10 - SIN(&
&           yaw(turb))*tempb9(nd)
          theta_c_0b(nd) = theta_c_0b(nd) + temp14*temp21*tempb18(nd)/&
&           14.7_dp + x0*tempb12(nd)
          x0b(nd) = x0b(nd) + theta_c_0*tempb12(nd)
          IF (1.0_dp - ct(turb) .EQ. 0.0) THEN
            ctb(nd, turb) = ctb(nd, turb) - tempb19(nd) - temp13*ky*kz*&
&             tempb9(nd)
          ELSE
            ctb(nd, turb) = ctb(nd, turb) + ((1.6_dp-temp32)/(2.0*temp35&
&             )-(1.6_dp*temp33+temp35)/(2.0*temp32))*tempb21(nd) + (&
&             temp28/(2.0*temp27)-(temp27+1.6_dp)/(2.0*temp31))*tempb20(&
&             nd) + ((-1.0)-1.3_dp/(2.0*temp25))*tempb19(nd) - temp13*ky&
&             *kz*tempb9(nd)
          END IF
          tempb14(nd) = kz*sigmazb(nd)
          tempb13(nd) = ky*sigmayb(nd)
          rotordiameterb(nd, turb) = rotordiameterb(nd, turb) + (temp12+&
&           temp21*temp24*temp26)*wake_offsetb(nd) - temp12*tempb12(nd) &
&           + temp30*2*rotordiameter(turb)*tempb10(nd) + temp34*2*&
&           rotordiameter(turb)*tempb11(nd) + (ky*temp9+COS(yaw(turb))/&
&           temp10)*sigmayb(nd) - temp9*tempb13(nd) - temp11*tempb14(nd)&
&           + (1.0/SQRT(8.0_dp)+kz*temp11)*sigmazb(nd) + ct(turb)*2*&
&           rotordiameter(turb)*tempb3(nd)
          deltax0b(nd) = tempb13(nd) + tempb14(nd)
        END DO
        CALL POPREAL4ARRAY(deltay, dp/4)
        CALL POPREAL4ARRAY(sigmaz, dp/4)
        CALL POPREAL4ARRAY(sigmay, dp/4)
      END IF
      CALL POPREAL4ARRAY(deltax0, dp/4)
      DO nd=1,nbdirs
        turbinexwb(nd, turbi) = turbinexwb(nd, turbi) + deltax0b(nd)
        turbinexwb(nd, turb) = turbinexwb(nd, turb) - deltax0b(nd)
        x0b(nd) = x0b(nd) - deltax0b(nd)
      END DO
    END DO
    CALL POPREAL4ARRAY(theta_c_0, dp/4)
    temp8 = COS(yaw(turb))
    temp5 = yaw(turb)/temp8
    temp7 = COS(yaw(turb))
    temp4 = -(ct(turb)*temp7) + 1.0_dp
    temp6 = SQRT(temp4)
    CALL POPREAL4ARRAY(x0, dp/4)
    temp3 = SQRT(-ct(turb) + 1.0_dp)
    temp2 = SQRT(2.0_dp)
    temp1 = temp2*(alpha*i+beta*(-temp3+1.0_dp))
    temp0 = SQRT(-ct(turb) + 1.0_dp)
    temp = COS(yaw(turb))
    DO nd=1,nbdirs
      tempb1(nd) = x0b(nd)/temp1
      IF (temp4 .EQ. 0.0) THEN
        tempb(nd) = 0.0
      ELSE
        tempb(nd) = -(temp5*0.3_dp*theta_c_0b(nd)/(2.0*temp6))
      END IF
      tempb0(nd) = (1.0_dp-temp6)*0.3_dp*theta_c_0b(nd)/temp8
      ctb(nd, turb) = ctb(nd, turb) - temp7*tempb(nd)
      yawb(nd, turb) = yawb(nd, turb) + (temp5*SIN(yaw(turb))+1.0)*&
&       tempb0(nd) - rotordiameter(turb)*(temp0+1.0_dp)*SIN(yaw(turb))*&
&       tempb1(nd) + ct(turb)*SIN(yaw(turb))*tempb(nd)
      rotordiameterb(nd, turb) = rotordiameterb(nd, turb) + (temp0+&
&       1.0_dp)*temp*tempb1(nd)
      IF (.NOT.1.0_dp - ct(turb) .EQ. 0.0) ctb(nd, turb) = ctb(nd, turb)&
&         + (-(beta*temp2*rotordiameter(turb)*temp*(temp0+1.0_dp)/(2.0*&
&         temp3*temp1))-rotordiameter(turb)*temp/(2.0*temp0))*tempb1(nd)
    END DO
  END DO
  DO nd=1,nbdirs
    yawdegb(nd, :) = 0.0
    yawdegb(nd, :) = -(pi*yawb(nd, :)/180.0_dp)
  END DO
  DO nd=1,nbdirs
    wtvelocityb(nd, :) = 0.0
  END DO
END SUBROUTINE PORTEAGEL_ANALYZE_BV
