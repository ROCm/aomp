! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
    integer :: var_a(10)
    !$omp declare target(var_a)
end module mod_a

PROGRAM prog_a
    use mod_a
    implicit none
    integer :: var_b

  !$omp target
    do var_b = 1, 10
        var_a(var_b) = var_b
    end do
  !$omp end target

  !$omp target
    do var_b = 1, 10
        var_a(var_b) = var_a(var_b) + var_b
    end do
  !$omp end target

  !$omp target update from(var_a)

   print *, var_a

   do var_b = 1, 10
    if (var_a(var_b) /= var_b + var_b) then
        print *, "======= FORTRAN Test Failed! ======="
    end if
   end do

   print *, "======= FORTRAN Test Passed! ======="
END PROGRAM
