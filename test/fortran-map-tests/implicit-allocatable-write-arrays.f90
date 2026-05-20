! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer, allocatable :: var_a(:)
    integer :: var_b
    allocate(var_a(10))

!$omp target
    do var_b = 1, 10
        var_a(var_b) = var_b
 end do
!$omp end target

 print *, var_a 

 do var_b = 1, 10
    if (var_a(var_b) /= var_b) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print *, "======= FORTRAN Test Passed! =======" 

end program
