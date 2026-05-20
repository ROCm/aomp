! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
        implicit none
        integer :: var_a(10), var_b = 0, var_c

        do var_c = 1, 10
                var_a(var_c) = 0
        end do
        call sub_a(var_a, 10)
        do var_c = 1, 10
                if ( var_a(var_c) /= var_c ) then
                        var_b = var_b + 1
                end if
        end do
        if ( var_b /= 0 ) then
                stop 1
        end if
        print*, "======= FORTRAN Test passed! ======="
end program prog_a
subroutine sub_a(var_a, var_b)
        implicit none
        integer :: var_a(*)
        integer :: var_b
        integer :: var_c
        integer :: var_d
        integer :: var_e

       var_d = 10
!$omp target map(tofrom:var_c)
        var_c = var_d
!$omp end target
        print*, var_c
        do var_e = 1, var_c
                var_a(var_e) = var_e
        end do
end subroutine sub_a
