! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none

  type :: dtype_a
    integer :: var_a
    real, allocatable :: var_b(:)
  end type dtype_a

  type :: dtype_b
    type(dtype_a) :: var_c
    integer, pointer :: var_d
    real, allocatable :: var_e
  end type dtype_b

  type :: dtype_c
    type(dtype_b), allocatable :: var_f
    type(dtype_a), pointer :: var_g
    type(dtype_a) :: var_h
    integer :: var_i
  end type dtype_c

  type(dtype_c), allocatable :: var_j
  integer, target :: var_k
  type(dtype_a), target :: var_l

  allocate(var_j)
  allocate(var_j%var_f)
  allocate(var_j%var_f%var_c%var_b(5))
  var_j%var_f%var_d => var_k
  allocate(var_j%var_f%var_e)
  allocate(var_l%var_b(3))
  var_j%var_g => var_l
  allocate(var_j%var_h%var_b(2))

  var_j%var_f%var_d = 5
  var_j%var_f%var_c%var_a = 10
  var_j%var_f%var_c%var_b(2) = 20

  !$omp target enter data map(to: var_j)

  var_j%var_f%var_d = 25
  var_j%var_f%var_c%var_a = 30
  var_j%var_f%var_c%var_b(2) = 40

  !$omp target update from(var_j)

  print *, "After target update from:"
  print *, " var_j%var_f%var_c%var_d = ", var_j%var_f%var_d
  print *, " var_j%var_f%var_c%var_a = ", var_j%var_f%var_c%var_a
  print *, " var_j%var_f%var_c%var_b(2) = ", var_j%var_f%var_c%var_b(2)

  if (var_j%var_f%var_d /= 25) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_j%var_f%var_c%var_a /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_j%var_f%var_c%var_b(2) /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  var_j%var_f%var_d = 25
  var_j%var_f%var_c%var_a = 30
  var_j%var_f%var_c%var_b(2) = 40

  !$omp target exit data map(from: var_j)

  print *, "After target exit data map:"
  print *, " var_j%var_f%var_c%var_d = ", var_j%var_f%var_d
  print *, " var_j%var_f%var_c%var_a = ", var_j%var_f%var_c%var_a
  print *, " var_j%var_f%var_c%var_b(2) = ", var_j%var_f%var_c%var_b(2)

  if (var_j%var_f%var_d /= 25) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_j%var_f%var_c%var_a /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_j%var_f%var_c%var_b(2) /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  deallocate(var_j%var_h%var_b)
  deallocate(var_l%var_b)
  deallocate(var_j%var_f%var_e)
  deallocate(var_j%var_f%var_c%var_b)
  deallocate(var_j%var_f)
  deallocate(var_j)

  print*, "======= FORTRAN Test Passed! ======="
end program prog_a
