! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
      real(4) :: var_a
      real(4) :: var_b
      integer, pointer :: var_c(:)
      real(4) :: var_d
    end type dtype_a

    type :: dtype_b
      integer(4) :: var_e
      integer(4) :: var_f
      integer, allocatable :: var_g
      integer(4) :: var_h
    end type dtype_b

    type :: dtype_c
     real(4) :: var_i(10)
     real(4) :: var_j
     integer, pointer :: var_k
     real(4) :: var_l(10)
     type(dtype_a) :: var_m
     type(dtype_b) :: var_n
    end type dtype_c

    type :: dtype_d
      real(4) :: var_o
      integer(4) :: var_p(10)
      real(4) :: var_q
      integer, allocatable :: var_r(:)
      integer(4) :: var_s
      type(dtype_c) :: var_t
    end type dtype_d

    type(dtype_d) :: var_u
    integer, target :: var_v(10)
    integer :: var_w

    var_u%var_t%var_m%var_c => var_v
    allocate(var_u%var_r(10))
    allocate(var_u%var_t%var_k)
    allocate(var_u%var_t%var_n%var_g)

    var_u%var_t%var_m%var_b = 12
!$omp target map(tofrom: var_u%var_t%var_m%var_b, var_u%var_t%var_m%var_a, var_u%var_t%var_n%var_g, var_u%var_t%var_m%var_d, var_u%var_t%var_m%var_c, var_u%var_p, var_u%var_t%var_k, var_u%var_t%var_n%var_e, var_u%var_t%var_j, var_u%var_t%var_n%var_h, var_u%var_r, var_u%var_t%var_n%var_f)
    var_u%var_t%var_m%var_a = 10
    var_u%var_t%var_m%var_b = 12 + var_u%var_t%var_m%var_b
    var_u%var_t%var_m%var_d = 54

    var_u%var_t%var_n%var_e = 20
    var_u%var_t%var_n%var_f = 40
    var_u%var_t%var_n%var_h = 60

    var_u%var_t%var_j = 200
    var_u%var_t%var_n%var_g = 30
    var_u%var_t%var_k = 200

    do var_w = 1, 10
      var_u%var_t%var_m%var_c(var_w) = var_w
      var_u%var_r(var_w) = var_w
      var_u%var_p(var_w) = var_w
    end do
!$omp end target

  print *, var_u%var_t%var_m%var_a
  print *, var_u%var_t%var_m%var_b
  print *, var_u%var_t%var_m%var_d
  print *, var_u%var_t%var_n%var_e
  print *, var_u%var_t%var_n%var_f
  print *, var_u%var_t%var_n%var_h
  print *, var_u%var_t%var_j
  print *, var_u%var_t%var_m%var_c
  print *, var_u%var_r
  print *, var_u%var_p
  print *, var_u%var_t%var_k
  print *, var_u%var_t%var_n%var_g

  if (var_u%var_t%var_m%var_a /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_m%var_b /= 24) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_m%var_d /= 54) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_n%var_e /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_n%var_f /= 40) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_n%var_h /= 60) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_j /= 200) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_k /= 200) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_u%var_t%var_n%var_g /= 30) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_w = 1, 10
    if (var_u%var_t%var_m%var_c(var_w) /= var_w) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  do var_w = 1, 10
    if (var_u%var_r(var_w) /= var_w) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  do var_w = 1, 10
    if (var_u%var_p(var_w) /= var_w) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN test passed! ======="
end program prog_a
