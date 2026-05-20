! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    integer(4), allocatable :: var_a
    allocate(var_a)
    var_a = 0

!$omp target map(tofrom:var_a)
    do var_b = 1, 10
      var_a = var_a + 50
    end do
!$omp end target

  print *, var_a

  if (var_a /= 500) then
     print *, "======= FORTRAN Test Failed! ======="
     stop 1
  end if

  deallocate(var_a)

  print *, "======= FORTRAN Test Passed! ======="
end program
