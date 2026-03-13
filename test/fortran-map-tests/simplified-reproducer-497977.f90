! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    use iso_fortran_env, only: real64
    implicit none
contains
    subroutine sub_a(var_a)
        implicit none
        integer :: var_b
        real(kind=real64), dimension(:) :: var_a
        real(kind=real64), dimension(size(var_a, 1)) :: var_c

!$omp target map(tofrom: var_c)
        do var_b = 1, 10
            var_c(var_b) = var_b
        end do
!$omp end target

        print *, var_c

        do var_b = 1, 10
            if (var_c(var_b) /= var_b) then
                print*, "======= FORTRAN Test Failed! ======="
                stop 1
            end if
        end do

        print *, "======= FORTRAN Test Passed! ======="
    end subroutine
end module mod_a

program prog_a
    use mod_a
    real(kind=real64), allocatable :: var_a(:)
    integer :: var_b
    allocate(var_a(10))

    do var_b = 1, 10
        var_a(var_b) = var_b
    end do

    call sub_a(var_a)
end program prog_a
