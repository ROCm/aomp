! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    save
    real(8), allocatable          ::   var_a(:,:)
    !$omp declare target(var_a)
end module mod_a

program prog_a
  use mod_a
  allocate(var_a(3,0:27647))

!$omp target enter data map(alloc: var_a)
!$omp target update from(var_a)
!$omp target update from(var_a)

  print *, "======= FORTRAN Test Passed! ======="
end program
