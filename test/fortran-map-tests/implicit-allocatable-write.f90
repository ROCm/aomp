! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer, allocatable :: var_a
    allocate(var_a)

!$omp target
    var_a = 1
!$omp end target

 print *, var_a 

if (var_a /= 1) then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
end if

  print *, "======= FORTRAN Test Passed! =======" 

end program
