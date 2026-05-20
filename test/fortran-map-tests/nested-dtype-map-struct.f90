! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
      real(8) :: var_a
      real(4) :: var_b(10)
      real(4) :: var_c(10)
    end type dtype_a

    type :: dtype_b
      real(4) :: var_d
      integer(4) :: var_e(10)
      real(4) :: var_f
      type(dtype_a) :: var_g
      integer, allocatable :: var_h(:)
      integer(4) :: var_i
      type(dtype_a) :: var_j
    end type dtype_b

    type(dtype_b) :: var_k
    integer :: var_l

!$omp target map(tofrom: var_k%var_i, var_k%var_j%var_b, var_k%var_g)
    do var_l = 1, 10
      var_k%var_j%var_b(var_l) = var_l * 2
      var_k%var_g%var_b(var_l) = var_l * 2
    end do

    var_k%var_g%var_a = 30.30
    var_k%var_i = 74
!$omp end target

  print *, var_k%var_g%var_a
  print *, var_k%var_i
  print *, var_k%var_g%var_b
  print *, var_k%var_j%var_b

  if (var_k%var_g%var_a /= 30.30) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  if (var_k%var_i /= 74) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_l = 1, 10
    if (var_k%var_g%var_b(var_l) /= var_l * 2) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  do var_l = 1, 10
    if (var_k%var_j%var_b(var_l) /= var_l * 2) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN test passed! ======="
end program prog_a
