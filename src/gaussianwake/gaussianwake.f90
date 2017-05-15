subroutine porteagel_analyze(nTurbines, turbineXw, turbineYw, turbineZ, &
                             rotorDiameter, Ct, axialInduction, wind_speed, &
                             yawDeg, ky, kz, alpha, beta, I, wtVelocity)

    ! independent variables: turbineXw turbineYw turbineZ rotorDiameter
    !                        yawDeg Ct turbineXw turbineYw rotorDiameter a_in

    ! dependent variables: wtVelocity

    implicit none

    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)

    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: turbineXw, turbineYw, turbineZ
    real(dp), dimension(nTurbines), intent(in) :: rotorDiameter, yawDeg
    real(dp), dimension(nTurbines), intent(in) :: Ct, axialInduction
    real(dp), intent(in) :: ky, kz, alpha, beta, I, wind_speed

    ! local (General)
    real(dp), dimension(nTurbines) :: yaw
    real(dp) :: x0, deltax0, deltay, theta_c_0, sigmay, sigmaz, wake_offset
    real(dp) :: deltav
    Integer :: turb, turbI
    real(dp), parameter :: pi = 3.141592653589793_dp

    ! model out
    real(dp), dimension(nTurbines), intent(out) :: wtVelocity

    intrinsic cos, atan, max, sqrt, log

    yaw = - yawDeg*pi/180.0_dp

    ! Glauert correction
    !do turb=1, nTurbines
    !  if (Ct(turb) > 0.96_dp)  ! Glauert condition
    !      axialInduction(turb) = 0.143_dp + sqrt(0.0203_dp - 0.6427_dp * (0.889_dp - Ct(turb)))
    !  else if
    !      axialInduction(turb) = 0.5_dp * (1.0_dp - sqrt(1.0_dp - Ct(turb)))
    !  end if
    !end do

    wtVelocity = wind_speed

    do, turb=1, nTurbines
        x0 = rotorDiameter(turb) * (cos(yaw(turb)) * (1.0_dp + sqrt(1.0_dp - Ct(turb))) / &
                                    (sqrt(2.0_dp) * (alpha * I + beta * (1.0_dp - sqrt(1.0_dp - Ct(turb))))))
        theta_c_0 = 0.3_dp * yaw(turb) * (1.0_dp - sqrt(1.0_dp - Ct(turb) * cos(yaw(turb)))) / cos(yaw(turb))

        do, turbI=1, nTurbines ! at turbineX-locations

            deltax0 = turbineXw(turbI) - (turbineXw(turb) + x0)

            if (deltax0 > 0.0_dp) then
                sigmay = rotorDiameter(turb) * (ky * deltax0 / rotorDiameter(turb) &
                                                + cos(yaw(turb)) / sqrt(8.0_dp))
                sigmaz = rotorDiameter(turb) * (kz * deltax0 / rotorDiameter(turb) &
                                                + 1.0_dp / sqrt(8.0_dp))
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

                deltay = turbineYw(turbI) - (turbineYw(turb) + wake_offset)

                deltav = wind_speed * (                                         &
                    (1.0_dp - sqrt(1.0_dp - Ct(turb) *                          &
                                   cos(yaw(turb)) / (8.0_dp * sigmay * sigmaz / &
                                                        (rotorDiameter(turb) ** 2)))) *  &
                    exp(-0.5_dp * ((deltay) / sigmay) ** 2) *                   &
                    exp(-0.5_dp * ((turbineZ(turbI) - turbineZ(turb)) / sigmaz) ** 2) &
                )

                wtVelocity(turbI) = wtVelocity(turbI) - deltav

            end if
        end do
    end do

    !print *, "fortran"

  end subroutine porteagel_analyze
