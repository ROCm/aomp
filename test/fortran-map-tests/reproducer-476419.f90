! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: var_a
    !$omp declare target(var_a)
  end module mod_a

  PROGRAM prog_a
  use mod_a

  ALLOCATE(var_a(10))

  !$omp target data map(alloc:var_a)
    DO var_b = 1, 2
      print *, "executing update"
      !$omp target update from(var_a)
    end do
  !$omp end target data
  print *, "======= FORTRAN Test Passed! ======="
END PROGRAM
