! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
contains
    subroutine sub_a(var_a)
        implicit none
        real, dimension(:), optional :: var_a
        integer :: var_b
       !$omp target data if(present(var_a)) map(alloc:var_a)
            do var_b = 1, 10
                var_a(var_b) = var_b
            end do
       !$omp end target data
    end subroutine sub_a
end module mod_a

program prog_a
    use mod_a
    real :: var_a(10)
    call sub_a(var_a)
    print *, var_a
end program prog_a
