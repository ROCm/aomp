! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    integer :: var_a
    var_a = 10

    !$omp target enter data map(to: var_a)

    var_a = 20

   !$omp target
      var_a = var_a + 50
   !$omp end target

   !$omp target exit data map(from: var_a)

   print *, var_a

   if (var_a /= 10) then
     print *, "======= FORTRAN Test Failed! ======="
     stop 1
   end if

   print*, "======= FORTRAN Test passed! ======="
end program
