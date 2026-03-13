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
     type(dtype_a) :: var_j
     type(dtype_b), allocatable :: var_k
    end type dtype_c

    type :: dtype_d
    real(4) :: var_l
    integer(4) :: var_m(10)
    real(4) :: var_n
    integer, allocatable :: var_o(:)
    integer(4) :: var_p
    type(dtype_c) :: var_q
    end type dtype_d

    type(dtype_d) :: var_r
    integer :: var_s

    allocate(var_r%var_o(10))
    allocate(var_r%var_q%var_k)
    allocate(var_r%var_q%var_k%var_f)
    allocate(var_r%var_q%var_j%var_b)

    var_r%var_q%var_j%var_b = 12

!$omp target map(tofrom: var_r%var_q%var_j%var_b, var_r%var_q%var_j%var_a) &
!$omp map(tofrom: var_r%var_q%var_j%var_c, var_r%var_m, var_r%var_q%var_k%var_d) &
!$omp map(tofrom: var_r%var_q%var_h, var_r%var_q%var_k%var_f, var_r%var_q%var_k%var_e) &
!$omp map(tofrom: var_r%var_o)
    var_r%var_q%var_j%var_a = 10
    var_r%var_q%var_j%var_b = 12 + var_r%var_q%var_j%var_b
    var_r%var_q%var_j%var_c = 54

    var_r%var_q%var_k%var_d = 20
    var_r%var_q%var_k%var_e = 40
    var_r%var_q%var_k%var_f = 60

    var_r%var_q%var_h = 200

    do var_s = 1, 10
      var_r%var_m(var_s) = var_s
    end do

    do var_s = 1, 10
      var_r%var_o(var_s) = var_s
    end do
!$omp end target

  print *, var_r%var_q%var_j%var_a
  print *, var_r%var_q%var_j%var_b
  print *, var_r%var_q%var_j%var_c

  print *, var_r%var_q%var_k%var_d
  print *, var_r%var_q%var_k%var_e
  print *, var_r%var_q%var_k%var_f

  print *, var_r%var_q%var_h

  print *, var_r%var_m

  print *, var_r%var_o

  if (var_r%var_q%var_j%var_a /= 10) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_r%var_q%var_j%var_b /= 24) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_r%var_q%var_j%var_c /= 54) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_r%var_q%var_k%var_d /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_r%var_q%var_k%var_e /= 40) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_r%var_q%var_k%var_f /= 60) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_r%var_q%var_h /= 200) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_s = 1, 10
    if (var_r%var_m(var_s) /= var_s) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  do var_s = 1, 10
    if (var_r%var_o(var_s) /= var_s) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN test passed! ======="
end program prog_a
