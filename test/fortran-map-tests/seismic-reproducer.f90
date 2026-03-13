! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    double precision, allocatable, dimension(:) :: var_a
end module mod_a

PROGRAM prog_a
    use mod_a
    allocate (var_a(5))

    !$omp target data map(alloc: var_a)

    !$omp target update from(var_a)

    !$omp end target data

      print *, "======= FORTRAN Test Passed! ======="
END PROGRAM
