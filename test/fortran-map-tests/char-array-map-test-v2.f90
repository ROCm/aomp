! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  TYPE dtype_a
    CHARACTER(LEN=16), DIMENSION(10,10) :: var_a
    CHARACTER(LEN=16),             DIMENSION(:,:), ALLOCATABLE :: var_b
    CHARACTER(LEN=16), DIMENSION(:), POINTER :: var_c
  END TYPE dtype_a
  type(dtype_a) :: var_d

!$OMP TARGET ENTER DATA MAP (ALLOC:var_d%var_a)

!$omp target map(tofrom: var_d%var_a)
    var_d%var_a(2,2) = 'c'
!$omp end target

!$OMP TARGET UPDATE FROM(var_d%var_a)

 print *, var_d%var_a(2,2)
 if (var_d%var_a(2,2) /= 'c') then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
 end if

  print *, "======= FORTRAN Test Passed! ======="
end program
