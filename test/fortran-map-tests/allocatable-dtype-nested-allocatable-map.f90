! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
        real(4) :: var_a
        integer(4) :: var_b(10)
        real(4) :: var_c
        integer, allocatable :: var_d(:)
        integer(4) :: var_e
    end type dtype_a

    type(dtype_a), allocatable :: var_f
    integer :: var_g

    allocate(var_f)
    allocate(var_f%var_d(10))

!$omp target map(tofrom: var_f%var_d)
    do var_g = 1, 10
        var_f%var_d(var_g) = var_g
    end do
!$omp end target

print *, var_f%var_d

do var_g = 1, 10
    if (var_f%var_d(var_g) /= var_g) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print*, "======= FORTRAN Test Passed! ======="
end program prog_a
