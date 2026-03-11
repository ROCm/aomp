! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
    real(4) :: var_a
    integer, allocatable :: var_b 
    integer(4) :: var_c(10)
    integer, allocatable :: var_d(:)
    integer(4) :: var_e
    end type dtype_a

    type :: dtype_b
    real(4) :: var_f
    integer, allocatable :: var_g 
    integer(4) :: var_h(10)
    real(4) :: var_i
    integer, allocatable :: var_j(:)
    integer(4) :: var_k
    type(dtype_a) :: var_l
    end type dtype_b

    type(dtype_b) :: var_m
    integer :: var_n

    allocate(var_m%var_l%var_d(10))
    allocate(var_m%var_l%var_b) 

    do var_n = 1, 10
        var_m%var_l%var_c(var_n) = var_n
    end do

    !$omp target map(tofrom: var_m%var_l%var_c, var_m%var_l%var_d)
    do var_n = 1, 10
        var_m%var_l%var_d(var_n) = var_m%var_l%var_c(var_n) + var_n
    end do
    !$omp end target

    print *, var_m%var_l%var_d
    print *, var_m%var_l%var_c

    do var_n = 1, 10
        if (var_m%var_l%var_d(var_n) /= var_n + var_n) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    do var_n = 1, 10
        if (var_m%var_l%var_c(var_n) /= var_n) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    print*, "======= FORTRAN Test Passed! ======="
end program prog_a
