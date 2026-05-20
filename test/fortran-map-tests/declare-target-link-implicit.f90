! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
    integer :: var_a = 10
    !$omp declare target link(var_a)
end module mod_a

program prog_a
use mod_a
implicit none

!$omp target
 var_a = 10 + 10
!$omp end target

 print *, var_a 

if (var_a /= 20) then
    print*, "======= FORTRAN Test Failed! ======="     
    stop 1    
end if  

print*, "======= FORTRAN Test passed! ======="

end program
