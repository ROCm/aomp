! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT


program prog_a
  implicit none
  integer,  pointer :: var_a(:)
  integer :: var_b, var_c

allocate(var_a(10))

!$omp target map(tofrom: var_a)
  do var_b = 1, 10
    var_a(var_b) = var_b
  end do
!$omp end target

  do var_b = 1, 10
    print*, var_a(var_b)
  end do

  do var_c = 1, 10
    if (var_a(var_c) /= var_c) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN Test Passed! ======="
end program
