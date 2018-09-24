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
    intrinsic acos, sqrt
    
    print *, turbine_y, turbine_z, rotor_diameter, &
                            wake_center_y, wake_center_z, wake_diameter, &
                            wake_overlap
    
    ! distance between wake center and rotor center
    if (wake_center_z .gt. turbine_z + tol or wake_center_z .lt. turbine_z - tol) then
        OVdYd = sqrt((wake_center_y-turbine_y)**2_dp + (wake_center_z - turbine_z)**2_dp)
    else
        OVdYd = abs(wake_center_y-turbine_y)
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
    if (OVdYd >= 0.0_dp + tol) then ! check case to avoid division by zero
!     if (OVdYd >= 0.0_dp) then ! check case to avoid division by zero
        OVL = (-OVr*OVr+OVRR*OVRR+OVdYd*OVdYd)/(2.0_dp*OVdYd)
    else
        OVL = 0.0_dp
    end if

    OVz = OVRR*OVRR-OVL*OVL

    ! Finish calculating the distance from the intersection line to the outer edge of the wake
    if (OVz > 0.0_dp + tol) then TODO
!     if (OVz > 0.0_dp) then
        OVz = sqrt(OVz)
    else
        OVz = 0.0_dp
    end if
    
    !print *, "OVRR, OVL, OVRR, OVr, OVdYd, OVz ", OVRR, OVL, OVRR, OVr, OVdYd, OVz
    
    

    if (OVdYd < (OVr+OVRR)) then ! if the rotor overlaps the wake
        !print *, "OVL: ", OVL
        if (OVL < OVRR .and. (OVdYd-OVL) < OVr) then
!         if (OVdYd > 0.0_dp + tol) then
!         if ((OVdYd > 0.0_dp) .and. (OVdYd > (OVRR - OVr))) then
            ! print *, "acos(OVL/OVRR), acos((OVdYd-OVL)/OVr), OVRR, OVL, OVr, OVdYd, OVL/OVRR, (OVdYd-OVL)/OVr ", &
!     & acos(OVL/OVRR), acos((OVdYd-OVL)/OVr), OVRR, OVL, OVr, OVdYd, OVL/OVRR, (OVdYd-OVL)/OVr
            wake_overlap = OVRR*OVRR*acos(OVL/OVRR) + OVr*OVr*acos((OVdYd-OVL)/OVr) - OVdYd*OVz
        else if (OVRR > OVr) then
            wake_overlap = pi*OVr*OVr
            !print *, "wake ovl: ", wake_overlap
        else
            wake_overlap = pi*OVRR*OVRR
        end if
    else
        wake_overlap = 0.0_dp
    end if
    print *, "wake overlap in func: ", wake_overlap/(pi*OVr**2)
    print *, "wake overlap in func: ", wake_overlap/(pi*OVRR**2)
    if ((wake_overlap/(pi*OVr**2) > 1.0_dp) .or. (wake_overlap/(pi*OVRR**2) > 1.0_dp)) then
        print *, "wake overlap in func: ", wake_overlap
        STOP 1
    end if
                             
end subroutine overlap_area_func