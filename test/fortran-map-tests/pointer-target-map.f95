! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT


program prog_a
implicit none
integer,  pointer :: var_a(:)
integer, target :: var_b(10)
integer :: var_c, var_d

var_a => var_b

!$omp target map(tofrom: var_a)
  do var_c = 1, 10
    var_a(var_c) = var_c
  end do
!$omp end target

  do var_c = 1, 10
    print*, var_a(var_c)
  end do

  do var_d = 1, 10
    if (var_a(var_d) /= var_d) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN Test Passed! ======="
end program
