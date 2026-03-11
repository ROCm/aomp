! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
  implicit none
  type dtype_a
    integer var_a
    real var_b
    real var_c(10)
  end type dtype_a

  type dtype_b
    character (LEN=30) var_d
    character (LEN=1) var_e
    type(dtype_a) var_f
  end type dtype_b

  type dtype_c
    integer var_g(20)
    type(dtype_b) var_h
    integer var_i
    type(dtype_a) var_j(5)
  end type dtype_c

  type(dtype_c) :: var_k(5)
  integer :: var_l

!$omp target map(tofrom: var_k(5))
    do var_l = 1, 20
      var_k(5)%var_g(var_l) = 20 + var_l
    end do

    var_k(5)%var_h%var_f%var_c(5) = 10
    var_k(5)%var_i = 40
!$omp end target

  do var_l = 1, 20
    print *, var_k(5)%var_g(var_l)
  end do

  print *, var_k(5)%var_g(5)
  print *, var_k(5)%var_h%var_f%var_c(5)
  print *, var_k(5)%var_i

  do var_l = 1, 20
    if (var_k(5)%var_g(var_l) /= 20 + var_l) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  if (var_k(5)%var_h%var_f%var_c(5) /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_k(5)%var_i /= 40) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

!$omp target map(tofrom: var_k(4)%var_g(3))
  var_k(4)%var_g(3) = 74
!$omp end target

 print *, var_k(4)%var_g(3)

if (var_k(4)%var_g(3) /= 74) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

!$omp target map(tofrom: var_k(5)%var_g(3:5))
  do var_l = 3, 5
     var_k(5)%var_g(var_l) = var_l
  end do
!$omp end target

 do var_l = 3, 5
    print *, var_k(5)%var_g(var_l)
 end do

 do var_l = 3, 5
  if (var_k(5)%var_g(var_l) /= var_l) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if
end do

!$omp target map(tofrom: var_k(3:5))
  do var_l = 3, 5
    var_k(var_l)%var_i = var_l
  end do
!$omp end target

  do var_l = 3, 5
    print *, var_k(var_l)%var_i
  end do

  do var_l = 3, 5
    if (var_k(var_l)%var_i /= var_l) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

!$omp target map(tofrom: var_k(4)%var_h%var_f%var_c(8))
  var_k(4)%var_h%var_f%var_c(8) = 200
!$omp end target

print *, var_k(4)%var_h%var_f%var_c(8)

if (var_k(4)%var_h%var_f%var_c(8) /= 200) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

!$omp target map(tofrom: var_k(4)%var_h%var_f%var_c(5:10))
  do var_l = 5, 10
    var_k(4)%var_h%var_f%var_c(var_l) = var_l
  end do
!$omp end target

do var_l = 5, 10
  print *, var_k(4)%var_h%var_f%var_c(var_l)
end do

do var_l = 5, 10
  if (var_k(4)%var_h%var_f%var_c(var_l) /= var_l) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
  end if
end do

!$omp target map(tofrom: var_k(4)%var_j(3)%var_c(4))
  var_k(4)%var_j(3)%var_c(4) = 200
!$omp end target

print *, var_k(4)%var_j(3)%var_c(4)

if (var_k(4)%var_j(3)%var_c(4) /= 200) then
  print*, "======= FORTRAN Test Failed! ======="
  stop 1
end if

print *, "======= FORTRAN Test Passed! ======="
end program prog_a
