! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

subroutine sub_a(var_a, var_b)
    implicit none
    integer :: var_b
    integer, dimension(var_b) :: var_a
    !$omp target enter data map(to:var_a)
end subroutine sub_a

subroutine sub_b(var_a, var_b)
    implicit none
    integer :: var_b
    integer, dimension(var_b) :: var_a
    !$omp target exit data map(from:var_a)
end subroutine sub_b

subroutine sub_c(var_a, var_b)
    implicit none
    integer :: var_b
    integer, dimension(var_b) :: var_a
    !$omp target update from(var_a)
end subroutine sub_c

program prog_a
    implicit none
    integer :: var_a = 10, var_b = 0
    integer :: var_c(10), var_d(10), var_e(10)

    call sub_a(var_c, var_a)
    call sub_a(var_d, var_a)

!$omp target
    do var_b = 1, 10
        var_c(var_b) = 10
    end do
!$omp end target

!$omp target
    do var_b = 1, 10
        var_d(var_b) = 20
    end do
!$omp end target

!$omp target map(from: var_e)
    do var_b = 1, 10
        var_e(var_b) = var_c(var_b) + var_d(var_b)
    end do
!$omp end target

    call sub_c(var_c, var_a)
    call sub_c(var_d, var_a)
    call sub_b(var_c, var_a)
    call sub_b(var_d, var_a)

    do var_b = 1, 10
        var_e(var_b) = var_e(var_b) + var_c(var_b) + var_d(var_b)
    end do

    print *, var_c
    print *, var_d
    print *, var_e

    do var_b = 1, 10
     if (var_e(var_b) /= 60) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
      end if
    end do

    print *, "======= FORTRAN Test Passed! ======="
end program prog_a
