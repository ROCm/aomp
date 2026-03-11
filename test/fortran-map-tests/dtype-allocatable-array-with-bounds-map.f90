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
    integer :: var_l, var_m

    allocate(var_k%var_h(10))
    allocate(var_k%var_e)

!$omp target map(tofrom: var_k%var_h(2:6))
    do var_m = 1, 10
        var_k%var_h(var_m) = var_m
    end do
!$omp end target

print *, var_k%var_h

if (var_k%var_h(1) == 1) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

do var_l = 2, 6
    if (var_k%var_h(var_l) /= var_l) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

do var_l = 7, 10
    if (var_k%var_h(var_l) == var_l) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print *, "======= FORTRAN Test Passed! ======="
end program prog_a
