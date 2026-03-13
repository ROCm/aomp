! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none

  integer :: var_a

  type dtype_a
  integer :: var_b
  real :: var_c
  real :: var_d(10)
  end type dtype_a

  type dtype_b
  character (LEN=30) :: var_e
  character (LEN=1) :: var_f
  type(dtype_a) :: var_g
  end type dtype_b

  type dtype_c
  integer :: var_h(10)
  type(dtype_b) :: var_i
  integer :: var_j
  type(dtype_a) :: var_k(3)
  end type dtype_c

  type (dtype_c) :: var_l

  do var_a = 1, 10
    var_l%var_h(var_a) = 0
  end do

!$omp target map(to: var_l) map(tofrom: var_l%var_i%var_g, var_l%var_j)
  var_l%var_j = 20
  do var_a = 1, 10
    var_l%var_h(var_a) = var_a
  end do
  var_l%var_i%var_g%var_c = 32.0
!$omp end target

print *, var_l%var_j
print *, var_l%var_i%var_g%var_c
print *, var_l%var_h

do var_a = 1, 10
  if (var_l%var_h(var_a) /= 0) then
    print *, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

if (var_l%var_j /= 20) then
  print *, "======= FORTRAN Test Failed! ======="
  stop 1
end if

if (var_l%var_i%var_g%var_c /= 32.0) then
  print *, "======= FORTRAN Test Failed! ======="
  stop 1
end if

print*, "======= FORTRAN Test passed! ======="

end program prog_a
