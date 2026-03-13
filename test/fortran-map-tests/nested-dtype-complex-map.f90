! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT


program prog_a
    implicit none
    type :: dtype_a
      real(8) :: var_a
      complex  :: var_b
      real(4) :: var_c(10)
      real(4) :: var_d(10)
    end type dtype_a

    type :: dtype_b
      real(4) :: var_e
      integer(4) :: var_f(10)
      real(4) :: var_g
      type(dtype_a) :: var_h
      integer, allocatable :: var_i(:)
      integer(4) :: var_j
      complex :: var_k
    end type dtype_b

    type(dtype_b) :: var_l
    integer :: var_m

!$omp target map(tofrom: var_l%var_h%var_a, var_l%var_j, var_l%var_h%var_b, var_l%var_h%var_c, var_l%var_k)
    do var_m = 1, 10
      var_l%var_h%var_c(var_m) = var_m * 2
    end do

    var_l%var_k = (10,20)
    var_l%var_h%var_b = (510,210)

    var_l%var_h%var_a = 30.30
    var_l%var_j = 74
!$omp end target

  print *, var_l%var_h%var_a
  print *, var_l%var_j
  print *, var_l%var_h%var_c
  print *, var_l%var_k
  print *, var_l%var_h%var_b

  if (var_l%var_h%var_a /= 30.30) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_l%var_j /= 74) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_l%var_k /= (10,20)) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_l%var_h%var_b /= (510,210)) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_l%var_j /= 74) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_m = 1, 10
    if (var_l%var_h%var_c(var_m) /= var_m * 2) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN test passed! ======="
end program prog_a
