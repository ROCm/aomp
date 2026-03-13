! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
        integer(4) :: var_a(10)
    end type dtype_a

    type(dtype_a) :: var_b
    integer :: var_c

    do var_c = 1, 10
        var_b%var_a(var_c) = var_c + var_c
    end do

    !$omp target enter data map(to: var_b%var_a(3:6))

    do var_c = 1, 10
        var_b%var_a(var_c) = 10
    end do

   !$omp target
    do var_c=3,6
        var_b%var_a(var_c) = var_b%var_a(var_c) + var_c
    end do
  !$omp end target

  !$omp target exit data map(from: var_b%var_a(3:6))

  print*, var_b%var_a

  do var_c = 1, 2
    if (var_b%var_a(var_c) /= 10) then
        print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_c = 3, 6
      if (var_b%var_a(var_c) /= var_c * 3) then
          print *, "======= FORTRAN Test Failed! ======="
        stop 1
      end if
  end do

  do var_c = 7, 10
    if (var_b%var_a(var_c) /= 10) then
        print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program
