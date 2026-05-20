! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    use iso_fortran_env, only: real64
    implicit none
contains
    subroutine sub_a(var_a, var_b, var_c, var_d, var_e)
        import
        implicit none
        real(kind=real64), dimension(:,:) :: var_a, var_b
        real(kind=real64) :: var_f(5)
        integer :: var_c, var_d, var_e
        integer :: var_g,var_h

        real(kind=real64), dimension(size(var_a, 1), size(var_a, 2)) :: var_i

        var_f(1) = 6
        var_f(2) = 8
        var_f(3) = 10
        var_f(4) = 12
        var_f(5) = 14

!$omp target data map(alloc:var_i)
    !$omp target teams
    !$omp distribute parallel do
        do var_h = var_c, var_d
            var_i(var_h,var_e) = var_a(var_h,var_e)+var_b(var_h,var_e)
        end do
    !$omp end target teams

    !$omp target update from(var_i)
!$omp end target data

       print *, var_i

       do var_h = var_c, var_d
           if (var_i(var_h,var_e) /= var_f(var_h)) then
               print*, "======= FORTRAN Test Failed! ======="
               stop 1
           end if
       end do

       print *, "======= FORTRAN Test Passed! ======="
    end subroutine
end module mod_a


program prog_a
    use mod_a
    implicit none
    integer :: var_a, var_b
    real(kind=real64), allocatable :: var_c(:,:), var_d(:,:)
    allocate(var_c(5,5))
    allocate(var_d(5,5))

    do var_a = 1, 5
        do var_b = 1, 5
            var_d(var_a,var_b) = var_a + var_b
            var_c(var_a,var_b) = var_a + var_b
        end do
    end do

    call sub_a(var_c, var_d, 1, 5, 2)
end program prog_a
