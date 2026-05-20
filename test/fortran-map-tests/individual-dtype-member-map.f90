! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT


program prog_a
    real :: var_a
    type :: dtype_a
        integer(4) :: var_b = 0
        real(4) :: var_c = 0.0
        complex(4) :: var_d = (0,0)
        real(4) :: var_e = 1.0
        type(dtype_a), allocatable :: var_f 
    end type dtype_a  
  
    type(dtype_a) :: var_g
    var_g%var_b = 10
    var_g%var_c = 2.0
    var_g%var_d = (2, 10)
    var_g%var_e = 12.0
    var_a = 21.0

  !$omp target map(from:var_g%var_c)
    var_g%var_c = var_a
  !$omp end target

   if (var_g%var_c /= 21.0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
   end if

   print*, "======= FORTRAN Test passed! ======="
  end program prog_a
