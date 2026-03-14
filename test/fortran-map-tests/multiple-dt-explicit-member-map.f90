! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  type :: dtype_a
    integer(4) :: var_a = 0
    real(4) :: var_b = 0.0
    complex(4) :: var_c = (0,0)
    type(dtype_a), allocatable :: var_d
    real(4) :: var_e = 1.0
  end type dtype_a

    type(dtype_a) :: var_f

  !$omp target map(from:var_f%var_b, var_f%var_e)
    var_f%var_b = 21.0
    var_f%var_e = 27.0
  !$omp end target

  print*, var_f%var_b
  print*, var_f%var_e

  if (var_f%var_b /= 21.0 .and. var_f%var_e /= 27.0) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
  end if

   print*, "======= FORTRAN Test passed! ======="
end program prog_a
