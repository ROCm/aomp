! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
        real(4) :: var_a
        real(4) :: var_b
        integer(4) :: var_c(10)
        integer(4) :: var_d
    end type dtype_a

    type :: dtype_b
        real(4) :: var_e
        integer(4) :: var_f(10)
        real(4) :: var_g
        type(dtype_a) :: var_h(10)
        integer(4) :: var_i
    end type dtype_b

    type(dtype_b) :: var_j
    integer :: var_k

!$omp target map(tofrom: var_j%var_h(4:8))
    do var_k = 4, 8
        var_j%var_h(var_k)%var_d = var_k
    end do
!$omp end target

do var_k = 1, 10
    print *, var_j%var_h(var_k)%var_d
end do

do var_k = 1, 3
    if (var_j%var_h(var_k)%var_d == var_k) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

do var_k = 4, 8
    if (var_j%var_h(var_k)%var_d /= var_k) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

do var_k = 9, 10
    if (var_j%var_h(var_k)%var_d == var_k) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print*, "======= FORTRAN Test Passed! ======="
end program prog_a
