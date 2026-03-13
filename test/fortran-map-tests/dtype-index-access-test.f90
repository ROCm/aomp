! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

PROGRAM prog_a
  implicit none

  type :: dtype_a
    integer, allocatable :: var_a
  end type dtype_a

  type :: dtype_b
   type(dtype_a), allocatable :: var_b(:)
  end type dtype_b

  type :: dtype_c
    integer(4) :: var_c = 0
    type(dtype_b) :: var_d(10)
    complex(4) :: var_e = (0,0)
    real(4) :: var_f = 1.0
  end type dtype_c

  type(dtype_c), allocatable :: var_g
  integer :: var_h, var_i

  allocate(var_g)
  allocate(var_g%var_d(1)%var_b(10))
  allocate(var_g%var_d(10)%var_b(10))

  allocate(var_g%var_d(1)%var_b(1)%var_a)
  allocate(var_g%var_d(10)%var_b(10)%var_a)

  var_g%var_d(1)%var_b(1)%var_a = 20
  var_g%var_d(10)%var_b(10)%var_a = 40

 !$omp target map(tofrom: var_g%var_d(1)%var_b(1)%var_a, var_h)
    var_h = var_g%var_d(1)%var_b(1)%var_a
 !$omp end target

 !$omp target map(tofrom: var_g%var_d(10)%var_b(10)%var_a, var_i)
    var_i = var_g%var_d(10)%var_b(10)%var_a
 !$omp end target

  print *, var_g%var_d(1)%var_b(1)%var_a
  print *, var_g%var_d(10)%var_b(10)%var_a
  print *, var_h
  print *, var_i

  if (var_h /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_i /= 40) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print*, "======= FORTRAN test passed! ======="
end program prog_a
