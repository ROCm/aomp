! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a

    type :: dtype_a
    real(4) :: var_a
    integer, allocatable :: var_b
    integer(4) :: var_c(10)
    real(4) :: var_d
    integer, allocatable :: var_e(:)
    integer(4) :: var_f
    end type dtype_a

    type(dtype_a) :: var_g
    integer :: var_h

    allocate(var_g%var_e(10))
    allocate(var_g%var_b)

    do var_h = 1, 10
        var_g%var_c(var_h) = var_h
    end do

    !$omp target map(to: var_g%var_c) map(from: var_g%var_e)
        do var_h = 1, 10
            var_g%var_e(var_h) = var_g%var_c(var_h) + var_h
        end do
    !$omp end target

    print *, var_g%var_c
    print *, var_g%var_e

    do var_h = 1, 10
        if (var_g%var_e(var_h) /= var_h + var_h) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    do var_h = 1, 10
        if (var_g%var_c(var_h) /= var_h) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    print*, "======= FORTRAN Test Passed! ======="
end program prog_a
