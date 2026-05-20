! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
        implicit none
        integer :: var_a = 5
    !$omp declare target link(var_a)
end module mod_a

program prog_a
        implicit none
        integer :: var_b
        integer :: var_c(10), var_d = 0
        do var_b = 1, 10
                var_c(var_b) = 0
        end do
        call sub_a(var_c, 10)
        do var_b = 1, 10
                if ( var_c(var_b) /= var_b ) then
                        var_d = var_d + 1
                end if
        end do
        if ( var_d /= 0 ) then
                print*, "======= FORTRAN Test Failed! ======="
                stop 1
        end if
        print*, "======= FORTRAN Test passed! ======="
end program prog_a
subroutine sub_a(var_b, var_c)
        use mod_a
        implicit none
        integer :: var_d
        integer :: var_b(*)
        integer :: var_c
        integer :: var_e
        integer :: var_f = 5

!$omp target map(tofrom:var_e) map(tofrom:var_a)
        var_e = var_a + var_f
!$omp end target
        print*, var_e
        do var_d = 1, var_e
                var_b(var_d) = var_d
        end do
end subroutine sub_a
