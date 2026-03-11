! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
implicit none

TYPE dtype_a
REAL,    DIMENSION(:,:), ALLOCATABLE :: var_a, var_b
INTEGER         :: var_c
REAL,    DIMENSION(:,:), ALLOCATABLE :: var_d
INTEGER         :: var_e
REAL,    DIMENSION(:,:), ALLOCATABLE :: var_f, var_g
END TYPE dtype_a

TYPE dtype_b
TYPE(dtype_a):: var_h
integer :: var_i
INTEGER :: var_j(4)
END TYPE dtype_b

TYPE dtype_c
INTEGER         :: var_k
TYPE(dtype_b), DIMENSION(:), ALLOCATABLE :: var_l
END TYPE dtype_c

end module mod_a

program prog_a
    use mod_a
  implicit none
  integer :: var_m, var_n, var_o
  TYPE(dtype_c)       :: var_p

  var_o = 0
  allocate(var_p%var_l(2))
  do var_m = 1, 2
      allocate(var_p%var_l(var_m)%var_h%var_a(2, 2))
      allocate(var_p%var_l(var_m)%var_h%var_b(2, 2))
      allocate(var_p%var_l(var_m)%var_h%var_d(2, 2))
      allocate(var_p%var_l(var_m)%var_h%var_f(2, 2))
      allocate(var_p%var_l(var_m)%var_h%var_g(2, 2))
      do var_n = 1, 4
          var_p%var_l(var_m)%var_j(var_n) = var_n * 10
      end do
  end do

var_p%var_l(1)%var_h%var_e = 30

print *, "Before: "
print *, var_p%var_l(1)%var_j
print *, var_p%var_l(2)%var_j

print *, var_p%var_l(1)%var_h%var_a(1, 1)
print *, var_p%var_l(1)%var_h%var_d(2, 2)
print *, var_p%var_l(1)%var_h%var_g(1, 2)
print *, var_p%var_l(1)%var_h%var_e
print *, var_o

!$omp target enter data map(alloc:     &
!$omp   var_p%var_l(1)%var_h%var_a, &
!$omp   var_p%var_l(1)%var_h%var_d, &
!$omp   var_p%var_l(1)%var_h%var_g, &
!$omp   var_p%var_l(1)%var_h%var_e)

!$omp target map(tofrom: var_o)
  var_p%var_l(1)%var_h%var_a(1, 1) = 20333
  var_p%var_l(1)%var_h%var_d(2, 2) = 20222
  var_p%var_l(1)%var_h%var_g(1, 2) = 20444
  var_p%var_l(1)%var_h%var_e = 512
  var_o = 10
!$omp end target

!$omp target exit data map(from:       &
!$omp   var_p%var_l(1)%var_h%var_a, &
!$omp   var_p%var_l(1)%var_h%var_d, &
!$omp   var_p%var_l(1)%var_h%var_g, &
!$omp   var_p%var_l(1)%var_h%var_e)

  print *, "After: "
  print *, var_p%var_l(1)%var_j
  print *, var_p%var_l(2)%var_j

  print *, var_p%var_l(1)%var_h%var_a(1, 1)
  print *, var_p%var_l(1)%var_h%var_d(2, 2)
  print *, var_p%var_l(1)%var_h%var_g(1, 2)
  print *, var_p%var_l(1)%var_h%var_e
  print *, var_o

  do var_m = 1, 2
    do var_n = 1, 4
      if (var_p%var_l(var_m)%var_j(var_n) /= var_n * 10) then
          print*, "======= FORTRAN Test Failed! ======="
          stop 1
      end if
    end do
  end do

  if (var_p%var_l(1)%var_h%var_a(1,1) /= 20333) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_p%var_l(1)%var_h%var_d(2, 2) /= 20222) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_p%var_l(1)%var_h%var_g(1, 2) /= 20444) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_p%var_l(1)%var_h%var_e /= 512) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_o /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print *, "======= FORTRAN Test Passed! ======="
end program prog_a
