! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer :: var_a(10)
    integer :: var_b

    do var_b = 1, 10
      var_a(var_b) = var_b + var_b
    end do

    !$omp target enter data map(to: var_a)

    do var_b = 1, 10
      var_a(var_b) = 10
    end do

   !$omp target
    do var_b=1,10
      var_a(var_b) = var_a(var_b) + var_b
    end do
  !$omp end target

  !$omp target exit data map(from: var_a)

  print*, var_a

  do var_b = 1, 10
      if (var_a(var_b) /= var_b * 3) then
          print *, "======= FORTRAN Test Failed! ======="
        stop 1
      end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program
