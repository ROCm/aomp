! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    integer, allocatable :: var_a
    allocate(var_a)

!$omp target map(tofrom:var_a)
     var_a = 50
!$omp end target

  print *, var_a

  if (var_a /= 50) then
     print *, "======= FORTRAN Test Failed! ======="
     stop 1
   end if

  print *, "======= FORTRAN Test Passed! ======="
end program
