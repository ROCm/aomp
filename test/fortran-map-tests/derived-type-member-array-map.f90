! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
      real(4) :: var_a
      real(4) :: var_b(10)
      real(4) :: var_c
      real(4) :: var_d(10)
      real(4) :: var_e
    end type dtype_a

    type(dtype_a) :: var_f
    integer :: var_g

  do var_g = 1, 10
    var_f%var_b(var_g) = var_g
  end do

  !$omp target map(tofrom:var_f%var_b, var_f%var_d)
    do var_g = 1, 10
      var_f%var_d(var_g) = var_f%var_b(var_g)
    end do
  !$omp end target

  print*, var_f%var_b
  print*, var_f%var_d

  do var_g = 1, 10
    if (var_f%var_d(var_g) /= var_g) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program prog_a
