! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
    INTEGER :: var_a = 0
!$omp declare target link(var_a)
end module mod_a

program prog_a
    use mod_a
    implicit none
!$omp target map(tofrom:var_a)
    var_a = 1
!$omp end target

PRINT *, var_a

if (var_a /= 1) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

print*, "======= FORTRAN Test passed! ======="
end program
