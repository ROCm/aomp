! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
        real(4) :: var_a
        real(4) :: var_b(10)
        real(4) :: var_c
        real(4) :: var_d(10)
        real(4) :: var_e
    end type dtype_a

    type(dtype_a) :: var_f
    type(dtype_a) :: var_g

  !$omp target map(tofrom:var_f%var_c, var_g%var_e)
    var_g%var_e = 10
    var_f%var_c = 15
  !$omp end target

  print*, var_f%var_c
  print*, var_g%var_e

  if (var_f%var_c /= 15) then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
  
  if (var_g%var_e /= 10) then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print*, "======= FORTRAN Test passed! ======="
end program prog_a
