! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
        real(4) :: var_a
        real(4) :: var_b
        integer(4) :: var_c(10)
        integer(4) :: var_d
    end type dtype_a

    type :: dtype_b
        real(4) :: var_e
        integer, allocatable :: var_f
        integer(4) :: var_g(10)
        type(dtype_a) :: var_h
        real(4) :: var_i
        integer, allocatable :: var_j(:)
        integer(4) :: var_k
    end type dtype_b

    type(dtype_b), allocatable :: var_l
    integer :: var_m

    allocate(var_l)

!$omp target map(tofrom: var_l%var_h%var_c, var_l%var_k)
    do var_m = 1, 10
        var_l%var_h%var_c(var_m) = var_m
    end do
    var_l%var_k = 50
!$omp end target

print *, var_l%var_k
print *, var_l%var_h%var_c

do var_m = 1, 10
    if (var_l%var_h%var_c(var_m) /= var_m) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

if (var_l%var_k /= 50) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

print*, "======= FORTRAN Test Passed! ======="
end program prog_a
