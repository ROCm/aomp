! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer,    pointer :: var_a
    integer,    target, allocatable :: var_b

    allocate(var_b)

    var_b = 30

!$omp target map(var_a)
    var_a => var_b
    var_a = 45
!$omp end target

    print *, var_b

    if (var_b /= 45) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print *, "======= FORTRAN Test Passed! ======="
end program prog_a
