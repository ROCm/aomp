! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer :: var_a
    type :: dtype_a
    real(4) :: var_b
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

    allocate(var_m%var_j(10))
    allocate(var_m%var_g)

    !$omp target map(tofrom: var_m%var_j, var_m%var_i)
        do var_a = 1, 10
            var_m%var_j(var_a) = var_a
        end do
        var_m%var_i = 50
    !$omp end target

    print *, var_m%var_i
    print *, var_m%var_j

    do var_a = 1, 10
        if (var_m%var_j(var_a) /= var_a) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    if (var_m%var_i /= 50) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
      end if

    print*, "======= FORTRAN Test Passed! ======="
end program prog_a
