! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none
  type :: dtype_a
    integer(4) :: var_a = 0
    real(4) :: var_b = 0.0
    complex(4) :: var_c = (0,0)
    integer(4) :: var_d(5)
  end type dtype_a

  type(dtype_a) :: var_e
  type(dtype_a) :: var_f

  integer :: var_g

  var_f%var_a = 10
  var_f%var_b = 2.0
  var_f%var_c = (2, 10)

  do var_g = 1, 5
    var_f%var_d(var_g) = var_g
  end do

!$omp target map(from:var_e)
  var_e%var_a = var_f%var_a
  var_e%var_b = var_f%var_b
  var_e%var_c = var_f%var_c

  do var_g = 1, 5
    var_e%var_d(var_g) = var_f%var_d(var_g)
  end do
!$omp end target

    print*, var_f%var_a
    print*, var_f%var_b
    write (*,*) var_f%var_c

    print*, var_e%var_a
    print*, var_e%var_b
    write (*,*)  var_e%var_c

  if (var_e%var_a /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_e%var_b /= 2.0) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_e%var_c /= (2, 10)) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_g = 1, 5
    if (var_e%var_d(var_g) /= var_g) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

 print*, "======= FORTRAN Test passed! ======="
end program prog_a
