! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
        real(4) :: var_a
        integer(4) :: var_b(10)
        integer(4) :: var_c
    end type dtype_a

    type :: dtype_b
        real(4) :: var_d
        integer, allocatable :: var_e
        integer(4) :: var_f(10)
        real(4) :: var_g
        integer, allocatable :: var_h(:)
        integer(4) :: var_i
        type(dtype_a) :: var_j
    end type dtype_b

    type(dtype_b) :: var_k
    integer :: var_l

    allocate(var_k%var_h(10))
    allocate(var_k%var_e)

    do var_l = 1, 10
        var_k%var_j%var_b(var_l) = var_l
    end do

    !$omp target map(tofrom: var_k%var_j, var_k%var_h)
    do var_l = 1, 10
        var_k%var_h(var_l) = var_k%var_j%var_b(var_l) + var_l
    end do
    !$omp end target

    print *, var_k%var_j%var_b
    print *, var_k%var_h

    do var_l = 1, 10
        if (var_k%var_h(var_l) /= var_l + var_l) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    do var_l = 1, 10
        if (var_k%var_j%var_b(var_l) /= var_l) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    print*, "======= FORTRAN Test Passed! ======="
end program prog_a
