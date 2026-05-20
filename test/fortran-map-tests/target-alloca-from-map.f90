! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
contains
    subroutine sub_a(var_a, var_b)
       implicit none 
       integer, allocatable :: var_a(:)
       integer :: var_b, var_c

      !$omp target map(from:var_a)
          do var_c = 1,var_b
                var_a(var_c) = var_c
         end do
       !$omp end target

         write (*,*) var_a

         do var_c = 1, var_b
            if (var_a(var_c) /= var_c) then
              print *, "======= FORTRAN Test Failed! ======="
              stop 1
            end if
          end do

          print *, "======= FORTRAN Test Passed! =======" 
    end subroutine
end module

program prog_a
    use mod_a
    implicit none
    integer, parameter :: var_a = 256
    integer, allocatable :: var_b(:)
    allocate(var_b(var_a))
    call sub_a(var_b, var_a)
end program
