! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
        real(4) :: var_a
        integer(4) :: var_b(10)
        integer(4) :: var_c
    end type dtype_a

    type :: dtype_b
        real(4) :: var_d
        integer(4) :: var_e(10)
        type(dtype_a) :: var_f
        integer, allocatable :: var_g(:)
        integer(4) :: var_h
    end type dtype_b

    type :: dtype_c
        real(4) :: var_i
        integer, allocatable :: var_j
        integer(4) :: var_k(10)
        real(4) :: var_l
        integer, allocatable :: var_m(:)
        integer(4) :: var_n
        type(dtype_b) :: var_o
    end type dtype_c

    type(dtype_c) :: var_p
    integer :: var_q, var_r

    allocate(var_p%var_o%var_g(10))

!$omp target map(tofrom: var_p%var_o%var_g(2:6))
    do var_q = 1, 10
        var_p%var_o%var_g(var_q) = var_q
    end do
!$omp end target

print *, var_p%var_o%var_g

if (var_p%var_o%var_g(1) == 1) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

do var_r = 2, 6
    if (var_p%var_o%var_g(var_r) /= var_r) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

do var_r = 7, 10
    if (var_p%var_o%var_g(var_r) == var_r) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print *, "======= FORTRAN Test Passed! ======="
end program prog_a
