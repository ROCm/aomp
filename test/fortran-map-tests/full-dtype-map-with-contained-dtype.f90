! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type dtype_a
       real(4) :: var_a(10)
    end type dtype_a

    type dtype_b
      type(dtype_a) :: var_b
    end type dtype_b

    type :: dtype_c
        real(4) :: var_c
        real(4) :: var_d(10)
        real(4) :: var_e
        type(dtype_b) :: var_f
        real(4) :: var_g
    end type dtype_c

    type(dtype_c) :: var_h
    integer :: var_i

  !$omp target map(tofrom:var_h)
    do var_i = 1, 10
      var_h%var_f%var_b%var_a(var_i) = var_i
    end do
  !$omp end target

  print*, var_h%var_f%var_b%var_a

  do var_i = 1, 10
    if (var_h%var_f%var_b%var_a(var_i) /= var_i) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program prog_a
