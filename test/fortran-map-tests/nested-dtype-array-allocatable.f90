! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
implicit none
type :: dtype_a
    real(4) :: var_a
    integer(4) :: var_b(10)
    integer, allocatable :: var_c(:)
    integer(4) :: var_d
end type dtype_a

type :: dtype_b
    real(4) :: var_e
    integer, allocatable :: var_f
    integer(4) :: var_g(10)
    real(4) :: var_h
    integer, allocatable :: var_i(:)
    integer(4) :: var_j
    type(dtype_a) :: var_k
end type dtype_b

type(dtype_b) :: var_l
integer :: var_m

allocate(var_l%var_k%var_c(10))

!$omp target map(tofrom: var_l%var_k%var_c)
    do var_m = 1, 10
        var_l%var_k%var_c(var_m) = var_m
    end do
!$omp end target

print *, var_l%var_k%var_c

do var_m = 1, 10
    if (var_l%var_k%var_c(var_m) /= var_m) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print*, "======= FORTRAN Test Passed! ======="
end program prog_a
