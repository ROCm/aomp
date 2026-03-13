! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    real(8), dimension(:), allocatable :: var_a
end module mod_a

PROGRAM prog_a
    use mod_a
    allocate (var_a(225))

    print *, "1"
    !$omp target enter data map(to:var_a)
    print *, "2"
    !$omp target enter data map(to:var_a)
    print *, "3"
    !$omp target
        var_a(1)=var_a(2)
    !$omp end target
    print *, "4"
    !$omp target exit data map(delete: var_a)
    print *, "5"
    print *, "======= FORTRAN Test Passed! ======="
END PROGRAM
