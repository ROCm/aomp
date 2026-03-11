! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  type :: dtype_a
  real(4) :: var_a
  real(4), allocatable :: var_b
  real(4) :: var_c
  end type dtype_a

  type :: dtype_b
    integer(4) :: var_d
    integer(4) :: var_e
    integer(4), allocatable :: var_f
  end type dtype_b

  type :: dtype_c
    real(4) :: var_g(10)
    real(4) :: var_h
    real(4) :: var_i(10)
  end type dtype_c

  type :: dtype_d
  real(4) :: var_j ! 4
  integer, allocatable :: var_k
  type(dtype_c) :: var_l
  type(dtype_c), allocatable :: var_m
  end type dtype_d

  type(dtype_d) :: var_n(5)
  type(dtype_d), allocatable :: var_o(:)

  allocate(var_n(1)%var_m)
  allocate(var_n(2)%var_m)
  allocate(var_n(3)%var_m)
  allocate(var_n(4)%var_m)
  allocate(var_n(5)%var_m)

  allocate(var_o(5))
  allocate(var_o(1)%var_m)
  allocate(var_o(2)%var_m)
  allocate(var_o(3)%var_m)
  allocate(var_o(4)%var_m)
  allocate(var_o(5)%var_m)

  allocate(var_o(1)%var_k)
  allocate(var_o(2)%var_k)
  allocate(var_o(3)%var_k)
  allocate(var_o(4)%var_k)
  allocate(var_o(5)%var_k)

!$omp target map(tofrom: var_n(1:3), var_o(1:3))
  var_o(2)%var_l%var_i(5) = 10
  var_n(2)%var_l%var_i(5) = 10
  var_o(2)%var_j = 10
  var_n(2)%var_j = 10
!$omp end target

  print *, var_o(2)%var_j
  print *, var_n(2)%var_j

  print *, var_o(2)%var_l%var_i(5)
  print *, var_n(2)%var_l%var_i(5)

  if (var_n(2)%var_j/= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_o(2)%var_j/= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_o(2)%var_l%var_i(5)/= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_n(2)%var_l%var_i(5)/= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print*, "======= FORTRAN test passed! ======="
end program prog_a
