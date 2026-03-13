! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    real(4), allocatable          ::   var_a(:)
    !$omp declare target(var_a)
end module mod_a

program prog_a
    use mod_a
    implicit none

    allocate(var_a(0:10))

    var_a(0) = 30.0
    var_a(1) = 40.0
    var_a(10) = 25.0

    !$omp target map(tofrom: var_a)
         var_a(0) = var_a(1)
    !$omp end target

    print *, var_a(0)
    print *, var_a(1)

    if (var_a(0) /= 40.0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_a(1) /= 40.0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print*, "======= FORTRAN Test Passed! ======="
end program
