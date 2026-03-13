! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    INTEGER :: var_a = 1
!$omp declare target link(var_a)
end module mod_a

program prog_a
use mod_a
implicit none
integer :: var_b

!$omp target map(tofrom:var_b) map(tofrom:var_a)
    var_b = var_a
!$omp end target

    PRINT *, var_b
    PRINT *, var_a

if (var_a /= var_b) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

print*, "======= FORTRAN Test passed! ======="
end program
