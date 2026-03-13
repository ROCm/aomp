! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none

  type :: dtype_a
    real(4) :: var_a
    integer, allocatable :: var_b
    integer(4) :: var_c(10)
    real(4) :: var_d
    integer, allocatable :: var_e(:)
    integer(4) :: var_f
  end type dtype_a

  type(dtype_a) :: var_g
  integer, allocatable :: var_h(:)
  integer :: var_i

  allocate(var_g%var_e(10))
  allocate(var_h(10))

!$omp target map(from: var_h)
  do var_i = 1, 10
      var_h(var_i) = var_i
  end do
!$omp end target

!$omp target map(from: var_g%var_e)
  do var_i = 1, 10
    var_g%var_e(var_i) = var_i
  end do
!$omp end target

  print *, var_h
  print *, var_g%var_e

  do var_i = 1, 10
    if (var_g%var_e(var_i) /= var_i) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  do var_i = 1, 10
    if (var_h(var_i) /= var_i) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print *, "======= FORTRAN Test Passed! ======="
end program prog_a
