! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    type :: dtype_a
    real(4) :: var_a
    real(4) :: var_b
    real(4), allocatable :: var_c
    end type dtype_a

    type :: dtype_b
      integer(4), allocatable :: var_d
      integer(4) :: var_e
      integer(4) :: var_f
    end type dtype_b

    type :: dtype_c
     real(4) :: var_g(10)
     type(dtype_a), allocatable :: var_h
     real(4) :: var_i
     real(4), allocatable :: var_j
     real(4) :: var_k(10)
     type(dtype_b), allocatable :: var_l
     integer(4), allocatable :: var_m
     type(dtype_b), allocatable :: var_n
    end type dtype_c

    type :: dtype_d
    real(4) :: var_o
    integer(4) :: var_p(10)
    real(4) :: var_q
    integer, allocatable :: var_r(:)
    integer(4) :: var_s
    type(dtype_c), allocatable :: var_t
    end type dtype_d

    type(dtype_d), allocatable :: var_u
    integer :: var_v

    allocate(var_u)
    allocate(var_u%var_t)
    allocate(var_u%var_t%var_h)
    allocate(var_u%var_t%var_l)
    allocate(var_u%var_t%var_n)
    allocate(var_u%var_t%var_h%var_c)
    allocate(var_u%var_t%var_l%var_d)
    allocate(var_u%var_t%var_j)
    allocate(var_u%var_t%var_m)
    allocate(var_u%var_r(10))

!$omp target map(tofrom: var_u%var_r, var_u%var_t%var_h%var_c, var_u%var_t%var_j) &
!$omp map(tofrom: var_u%var_t%var_l%var_d, var_u%var_t%var_m, var_u%var_t%var_n)
    var_u%var_t%var_h%var_c = 54
    var_u%var_t%var_l%var_d = 20
    var_u%var_t%var_j = 104
    var_u%var_t%var_m = 204
    do var_v = 1, 10
      var_u%var_r(var_v) = var_v
    end do
    var_u%var_t%var_n%var_e = 10
!$omp end target

  print *, var_u%var_t%var_l%var_d
  print *, var_u%var_t%var_h%var_c
  print *, var_u%var_t%var_m
  print *, var_u%var_t%var_j
  print *, var_u%var_t%var_n%var_e
  print *, var_u%var_r

  if (var_u%var_t%var_h%var_c /= 54) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_l%var_d /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_j /= 104) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_m /= 204) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_n%var_e /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_v = 1, 10
    if (var_u%var_r(var_v) /= var_v) then
      print*, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  print*, "======= FORTRAN test passed! ======="
end program prog_a
