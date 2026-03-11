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
    type(dtype_a) :: var_g
    integer :: var_h

  !$omp target map(tofrom:var_f%var_b(3:6), var_g%var_b(3:6))
    do var_h = 1, 10
      var_g%var_b(var_h) = var_h
      var_f%var_b(var_h) = var_h
    end do
  !$omp end target

  print*, var_f%var_b  
  print*, var_g%var_b

  do var_h = 1, 2
    if (var_g%var_b(var_h) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_h = 3, 6
    if (var_g%var_b(var_h) /= var_h) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_h = 7, 10
    if (var_g%var_b(var_h) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_h = 1, 2
    if (var_f%var_b(var_h) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_h = 3, 6
    if (var_f%var_b(var_h) /= var_h) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_h = 7, 10
    if (var_f%var_b(var_h) /= 0) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print*, "======= FORTRAN Test passed! ======="
end program prog_a
