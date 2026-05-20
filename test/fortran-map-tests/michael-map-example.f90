! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer, parameter :: var_a = 100
    real, parameter :: var_b=0.2

    type :: dtype_a
      real, dimension(var_a) :: var_c
      real, dimension(var_a) :: var_d
     end type dtype_a

    type :: dtype_b
      real, dimension(var_a) :: var_e
     end type dtype_b

    type(dtype_a) :: var_f,var_g
    type(dtype_b) :: var_h

    integer :: var_i

    do var_i = 1, var_a
      var_f%var_c(var_i) = var_i-1
      var_g%var_c(var_i) = 1.0
    end do

    !$omp target data map(to: var_f%var_c, var_g, var_g%var_c) map(from: var_h%var_e)
    !$omp target
    do var_i = 1, var_a
      var_h%var_e(var_i) =  var_f%var_c(var_i) * var_g%var_c(var_i)
    end do
    !$omp end target
    !$omp end target data

    write(*,*) 'Done! ', var_h%var_e(var_a-1)

    if (var_h%var_e(var_a-1) /= 98) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if

   print*, "======= FORTRAN Test Passed! ======="
end program prog_a
