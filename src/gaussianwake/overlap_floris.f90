subroutine calcOverlapAreas(nTurbines, turbineX, turbineY, rotorDiameter, wakeDiameters, &
                            wakeCenters, wakeOverlapTRel_mat)
!    calculate overlap of rotors and wake zones (wake zone location defined by wake 
!    center and wake diameter)
!   turbineX,turbineY is x,y-location of center of rotor
!
!    wakeOverlap(TURBI,TURB,ZONEI) = overlap area of zone ZONEI of wake of turbine 
!     TURB with rotor of downstream turbine
!    TURBI

    implicit none
        
    ! define precision to be the standard for a double precision ! on local system
    integer, parameter :: dp = kind(0.d0)
    
    ! in
    integer, intent(in) :: nTurbines
    real(dp), dimension(nTurbines), intent(in) :: turbineX, turbineY, rotorDiameter
    real(dp), dimension(nTurbines, nTurbines, 3), intent(in) :: wakeDiameters
    real(dp), dimension(nTurbines, nTurbines), intent(in) :: wakeCenters
    
    ! out    
    real(dp), dimension(nTurbines, nTurbines, 3), intent(out) :: wakeOverlapTRel_mat
    
    ! local
    integer :: turb, turbI, zone
    real(dp), parameter :: pi = 3.141592653589793_dp, tol = 0.000001_dp
    real(dp) :: OVdYd, OVr, OVRR, OVL, OVz
    real(dp), dimension(nTurbines, nTurbines, 3) :: wakeOverlap
        
    wakeOverlapTRel_mat = 0.0_dp
    wakeOverlap = 0.0_dp
    
    OVdYd = wakeCenters(turbI, turb)-turbineY(turbI)    ! distance between wake center and rotor center
    OVr = rotorDiameter(turbI)/2                        ! rotor diameter
    
    OVRR = wakeDiameters(turbI, turb, zone)/2.0_dp        ! wake diameter
    OVdYd = abs(OVdYd)
    if (OVdYd >= 0.0_dp + tol) then
        ! calculate the distance from the wake center to the vertical line between
        ! the two circle intersection points
        OVL = (-OVr*OVr+OVRR*OVRR+OVdYd*OVdYd)/(2.0_dp*OVdYd)
    else
        OVL = 0.0_dp
    end if

    OVz = OVRR*OVRR-OVL*OVL

    ! Finish calculating the distance from the intersection line to the outer edge of the wake zone
    if (OVz > 0.0_dp + tol) then
        OVz = sqrt(OVz)
    else
        OVz = 0.0_dp
    end if

    if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake zone

        if (OVL < OVRR .and. (OVdYd-OVL) < OVr) then
            wakeOverlap(turbI, turb, zone) = OVRR*OVRR*dacos(OVL/OVRR) + OVr*OVr*dacos((OVdYd-OVL)/OVr) - OVdYd*OVz
        else if (OVRR > OVr) then
            wakeOverlap(turbI, turb, zone) = pi*OVr*OVr
        else
            wakeOverlap(turbI, turb, zone) = pi*OVRR*OVRR
        end if
    else
        wakeOverlap(turbI, turb, zone) = 0.0_dp
    end if


    do turb = 1, nTurbines
    
        do turbI = 1, nTurbines
    
            wakeOverlap(turbI, turb, 3) = wakeOverlap(turbI, turb, 3)-wakeOverlap(turbI, turb, 2)
            wakeOverlap(turbI, turb, 2) = wakeOverlap(turbI, turb, 2)-wakeOverlap(turbI, turb, 1)
    
        end do
    
    end do
    
    wakeOverlapTRel_mat = wakeOverlap

    do turbI = 1, nTurbines
            wakeOverlapTRel_mat(turbI, :, :) = wakeOverlapTRel_mat(turbI, :, &
                                                         :)/((pi*rotorDiameter(turbI) &
                                                       *rotorDiameter(turbI))/4.0_dp)
    end do
    
    ! do turbI = 1, nTurbines
!         do turb = 1, nTurbines
!             do zone = 1, 3
!                 print *, "wakeOverlapTRel_mat[", turbI, ", ", turb, ", ", zone, "] = ", wakeOverlapTRel_mat(turbI, turb, zone)
!             end do
!         end do
!     end do
        
   
                                    
end subroutine calcOverlapAreas