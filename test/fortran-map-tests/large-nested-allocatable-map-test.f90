! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none
  type dtype_a
    integer, allocatable :: var_a
    real, allocatable :: var_b
    real, allocatable :: var_c(:)
  end type dtype_a

  type dtype_b
    character (LEN=30), allocatable :: var_d
    character (LEN=1), allocatable :: var_e
    type(dtype_a), allocatable :: var_f
  end type dtype_b

  type dtype_c
    integer, allocatable :: var_g(:)
    type(dtype_b), allocatable :: var_h
    integer, allocatable :: var_i
    type(dtype_a), allocatable :: var_j(:)
  end type dtype_c

  type (dtype_c), allocatable :: var_k
  type (dtype_c), allocatable :: var_l
  integer :: var_m, var_n

  allocate(var_k)
  allocate(var_k%var_g(20))
  allocate(var_k%var_h)
  allocate(var_k%var_h%var_d)
  allocate(var_k%var_h%var_e)
  allocate(var_k%var_h%var_f)
  allocate(var_k%var_h%var_f%var_a)
  allocate(var_k%var_h%var_f%var_b)
  allocate(var_k%var_h%var_f%var_c(10))
  allocate(var_k%var_i)
  allocate(var_k%var_j(5))
  do var_n = 1, 5
    allocate(var_k%var_j(var_n)%var_a)
    allocate(var_k%var_j(var_n)%var_b)
    allocate(var_k%var_j(var_n)%var_c(10))
  end do

  allocate(var_l)
  allocate(var_l%var_g(20))
  allocate(var_l%var_h)
  allocate(var_l%var_h%var_d)
  allocate(var_l%var_h%var_e)
  allocate(var_l%var_h%var_f)
  allocate(var_l%var_h%var_f%var_a)
  allocate(var_l%var_h%var_f%var_b)
  allocate(var_l%var_h%var_f%var_c(10))
  allocate(var_l%var_i)
  allocate(var_l%var_j(5))
  do var_n = 1, 5
    allocate(var_l%var_j(var_n)%var_a)
    allocate(var_l%var_j(var_n)%var_b)
    allocate(var_l%var_j(var_n)%var_c(10))
  end do

  !$omp target map(tofrom: var_k%var_h%var_f%var_a, var_k%var_h%var_f%var_b, var_k%var_h%var_f%var_c, var_k%var_i, var_k%var_g, var_k%var_h%var_e, var_l%var_h%var_f%var_a, var_l%var_h%var_f%var_b, var_l%var_h%var_f%var_c, var_l%var_i, var_l%var_g, var_l%var_h%var_e)
  var_k%var_h%var_f%var_a = 5
  var_k%var_h%var_f%var_b = 10

  do var_m = 1, 10
    var_k%var_h%var_f%var_c(var_m) = 50
  end do

  var_k%var_i = 20

  do var_m = 1, 20
    var_k%var_g(var_m) = var_m
  end do

  var_k%var_h%var_e = "c"

  var_l%var_h%var_f%var_a = 5
  var_l%var_h%var_f%var_b = 10

  do var_m = 1, 10
    var_l%var_h%var_f%var_c(var_m) = 50
  end do

  var_l%var_i = 20

  do var_m = 1, 20
    var_l%var_g(var_m) = var_m
  end do

  var_l%var_h%var_e = "c"
!$omp end target

print *, var_k%var_h%var_e
print *, var_k%var_i
print *, var_k%var_h%var_f%var_a
print *, var_k%var_h%var_f%var_b

do var_m = 1, 20
  print *, var_k%var_g(var_m)
end do

do var_m = 1, 10
  print *, var_k%var_h%var_f%var_c(var_m)
end do

if (var_k%var_h%var_e /= "c") then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_k%var_i /= 20) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_k%var_h%var_f%var_a /= 5) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_k%var_h%var_f%var_b /= 10) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

do var_m = 1, 20
  if (var_k%var_g(var_m) /= var_m) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

do var_m = 1, 10
  if (var_k%var_h%var_f%var_c(var_m) /= 50) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

print *, var_l%var_h%var_e
print *, var_l%var_i
print *, var_l%var_h%var_f%var_a
print *, var_l%var_h%var_f%var_b

do var_m = 1, 20
  print *, var_l%var_g(var_m)
end do

do var_m = 1, 10
  print *, var_l%var_h%var_f%var_c(var_m)
end do

if (var_l%var_h%var_e /= "c") then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_l%var_i /= 20) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_l%var_h%var_f%var_a /= 5) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_l%var_h%var_f%var_b /= 10) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

do var_m = 1, 20
  if (var_l%var_g(var_m) /= var_m) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

do var_m = 1, 10
  if (var_l%var_h%var_f%var_c(var_m) /= 50) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

print *, "======= FORTRAN Test Passed! ======="
end program prog_a
