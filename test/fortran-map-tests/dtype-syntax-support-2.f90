! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none

    integer :: var_a
    integer :: var_b, var_c

    type :: dtype_a
        integer :: var_d
        integer :: var_e(10)
        integer(4), allocatable :: var_f(:)
        integer(4), allocatable :: var_g(:)
    end type dtype_a

    type dtype_b
        TYPE(dtype_a) :: var_h
    end type dtype_b

    type :: dtype_c
        real(4) :: var_i
        type(dtype_a), allocatable :: var_j(:)
        TYPE(dtype_b), DIMENSION(:), allocatable :: var_k
        integer(4) :: var_l(10)
        real(4) :: var_m
        integer, allocatable :: var_n(:)
        integer(4) :: var_o
    end type dtype_c

    type(dtype_c) :: var_p
    type(dtype_c), DIMENSION(:), allocatable :: var_q

    integer :: var_r(10,10,10)

    allocate(var_p%var_j(4))
    allocate(var_p%var_j(1)%var_f(10))
    allocate(var_p%var_j(1)%var_g(10))
    allocate(var_p%var_j(2)%var_f(10))
    allocate(var_p%var_j(2)%var_g(10))
    allocate(var_p%var_j(3)%var_f(10))
    allocate(var_p%var_j(3)%var_g(10))
    allocate(var_p%var_j(4)%var_f(10))
    allocate(var_p%var_j(4)%var_g(10))
    allocate(var_p%var_k(4))
    allocate(var_p%var_k(1)%var_h%var_f(10))
    allocate(var_p%var_k(1)%var_h%var_g(10))
    allocate(var_p%var_k(2)%var_h%var_f(10))
    allocate(var_p%var_k(2)%var_h%var_g(10))
    allocate(var_p%var_k(3)%var_h%var_f(10))
    allocate(var_p%var_k(3)%var_h%var_g(10))
    allocate(var_p%var_k(4)%var_h%var_f(10))
    allocate(var_p%var_k(4)%var_h%var_g(10))

    allocate(var_q(3))

    do var_a = 1, 3
        allocate(var_q(var_a)%var_j(4))
        allocate(var_q(var_a)%var_j(1)%var_f(10))
        allocate(var_q(var_a)%var_j(1)%var_g(10))
        allocate(var_q(var_a)%var_j(2)%var_f(10))
        allocate(var_q(var_a)%var_j(2)%var_g(10))
        allocate(var_q(var_a)%var_j(3)%var_f(10))
        allocate(var_q(var_a)%var_j(3)%var_g(10))
        allocate(var_q(var_a)%var_j(4)%var_f(10))
        allocate(var_q(var_a)%var_j(4)%var_g(10))
        allocate(var_q(var_a)%var_k(4))
        allocate(var_q(var_a)%var_k(1)%var_h%var_f(10))
        allocate(var_q(var_a)%var_k(1)%var_h%var_g(10))
        allocate(var_q(var_a)%var_k(2)%var_h%var_f(10))
        allocate(var_q(var_a)%var_k(2)%var_h%var_g(10))
        allocate(var_q(var_a)%var_k(3)%var_h%var_f(10))
        allocate(var_q(var_a)%var_k(3)%var_h%var_g(10))
        allocate(var_q(var_a)%var_k(4)%var_h%var_f(10))
        allocate(var_q(var_a)%var_k(4)%var_h%var_g(10))
    end do

    var_b = 1
    var_c = 2

!$omp target map(tofrom: var_p%var_j(var_b)%var_d)
        var_p%var_j(var_b)%var_d = 3
!$omp end target

print *, var_p%var_j(var_b)%var_d

if (var_p%var_j(var_b)%var_d /= 3) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

!$omp target map(tofrom: var_p%var_j(var_b)%var_d, var_p%var_j(var_c)%var_d)
        var_p%var_j(var_b)%var_d = 5
        var_p%var_j(var_c)%var_d = 10
!$omp end target

do var_a = 1, 4
    print *, var_p%var_j(var_a)%var_d
end do

if (var_p%var_j(var_b)%var_d /= 5) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

if (var_p%var_j(var_c)%var_d /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

!$omp target map(tofrom: var_p%var_k(var_b)%var_h%var_f, &
!$omp                    var_p%var_k(var_b)%var_h%var_g)
    do var_a = 1, 10
        var_p%var_k(var_b)%var_h%var_f(var_a) = var_a
        var_p%var_k(var_b)%var_h%var_g(var_a) = var_a
    end do
!$omp end target

print *, var_p%var_k(var_b)%var_h%var_f
print *, var_p%var_k(var_b)%var_h%var_g

do var_a = 1, 10
    if (var_p%var_k(var_b)%var_h%var_f(var_a) /= var_a) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if

    if (var_p%var_k(var_b)%var_h%var_g(var_a) /= var_a) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

!$omp target map(tofrom:  var_p%var_k(var_b)%var_h%var_d, &
!$omp                     var_p%var_k(var_c)%var_h%var_d, &
!$omp                     var_p%var_k(var_b)%var_h%var_g, &
!$omp                     var_p%var_k(var_c)%var_h%var_g)
    var_p%var_k(var_c)%var_h%var_d = 9999
    var_p%var_k(var_c)%var_h%var_g(2) = 9998
    var_p%var_k(var_b)%var_h%var_d = 9997
    var_p%var_k(var_b)%var_h%var_g(2) = 9996
!$omp end target

print *, var_p%var_k(var_b)%var_h%var_d
print *, var_p%var_k(var_c)%var_h%var_d
print *, var_p%var_k(var_b)%var_h%var_g(2)
print *, var_p%var_k(var_c)%var_h%var_g(2)

if (var_p%var_k(var_b)%var_h%var_d /= 9997) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

if (var_p%var_k(var_c)%var_h%var_d /= 9999) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

if (var_p%var_k(var_b)%var_h%var_g(2) /= 9996) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

if (var_p%var_k(var_c)%var_h%var_g(2) /= 9998) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

!$omp target map(tofrom:  var_p%var_k(var_c)%var_h%var_g)
   var_p%var_k(var_c)%var_h%var_g(2) = 2000
!$omp end target

if (var_p%var_k(var_c)%var_h%var_g(2) /= 2000) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

!$omp target map(tofrom: var_p%var_j(var_b)%var_f, &
!$omp                    var_p%var_j(var_b)%var_g, &
!$omp                    var_p%var_j(var_c)%var_f, &
!$omp                    var_p%var_j(var_c)%var_g)
    do var_a = 1, 10
        var_p%var_j(var_b)%var_f(var_a) = var_a * 2
        var_p%var_j(var_b)%var_g(var_a) = var_a * 2
        var_p%var_j(var_c)%var_f(var_a) = var_a * 2
        var_p%var_j(var_c)%var_g(var_a) = var_a * 2
    end do
!$omp end target

print *, var_p%var_j(var_b)%var_f
print *, var_p%var_j(var_b)%var_g
print *, var_p%var_j(var_c)%var_f
print *, var_p%var_j(var_c)%var_g

do var_a = 1, 10
    if (var_p%var_j(var_b)%var_f(var_a) /= var_a * 2) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if

    if (var_p%var_j(var_b)%var_g(var_a) /= var_a * 2) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_p%var_j(var_c)%var_f(var_a) /= var_a * 2) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
      end if

      if (var_p%var_j(var_c)%var_g(var_a) /= var_a * 2) then
          print*, "======= FORTRAN Test Failed! ======="
          stop 1
      end if
end do

!$omp target map(tofrom: var_p%var_j(var_b)%var_f, &
!$omp                    var_p%var_j(var_b)%var_g, &
!$omp                    var_p%var_j(4)%var_g, &
!$omp                    var_p%var_j(4)%var_f, &
!$omp                    var_p%var_j(var_c)%var_f, &
!$omp                    var_p%var_j(var_c)%var_g)
    do var_a = 1, 10
        var_p%var_j(var_b)%var_f(var_a) = var_a * 3
        var_p%var_j(var_b)%var_g(var_a) = var_a * 3
        var_p%var_j(4)%var_f(var_a) = var_a * 3
        var_p%var_j(4)%var_g(var_a) = var_a * 3
        var_p%var_j(var_c)%var_f(var_a) = var_a * 3
        var_p%var_j(var_c)%var_g(var_a) = var_a * 3
    end do
!$omp end target


    print *, var_p%var_j(1)%var_f
    print *, var_p%var_j(1)%var_g
    print *, var_p%var_j(4)%var_f
    print *, var_p%var_j(4)%var_g
    print *, var_p%var_j(2)%var_f
    print *, var_p%var_j(2)%var_g

do var_a = 1, 10
    if (var_p%var_j(1)%var_f(var_a) /= var_a * 3) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
        end if

    if (var_p%var_j(1)%var_g(var_a) /= var_a * 3) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_p%var_j(4)%var_f(var_a) /= var_a * 3) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_p%var_j(4)%var_g(var_a) /= var_a * 3) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_p%var_j(2)%var_f(var_a) /= var_a * 3) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
        end if

    if (var_p%var_j(2)%var_g(var_a) /= var_a * 3) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

!$omp target map(tofrom: var_q(var_c)%var_l)
    do var_a = 1, 10
        var_q(var_c)%var_l(var_a) = var_a
    end do
!$omp end target

print *, var_q(var_c)%var_l

do var_a = 1, 10
    if (var_q(var_c)%var_l(var_a) /= var_a) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
end do

print *, "======= FORTRAN Test Passed! ======="

end program prog_a
