! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
        real(4) :: var_a
        real(4), allocatable :: var_b
        integer(4) :: var_c(10)
        integer(4) :: var_d
        integer(4) :: var_e
        real(4), allocatable :: var_f
        integer(4) :: var_g
    end type dtype_a

    type :: dtype_b
        real(4) :: var_h
        real(4), allocatable :: var_i
        integer(4) :: var_j(10)
        real(4) :: var_k
        type(dtype_a), allocatable :: var_l
        integer(4) :: var_m
    end type dtype_b


    type(dtype_b), allocatable :: var_n
    allocate(var_n)
    allocate(var_n%var_l)
    allocate(var_n%var_l%var_b)

!$omp target map(tofrom: var_n%var_l%var_b)
    var_n%var_l%var_b = 100
!$omp end target

print *, var_n%var_l%var_b

if (var_n%var_l%var_b /= 100) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

print*, "======= FORTRAN Test Passed! ======="
end program prog_a
