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
    end type dtype_b

    type(dtype_b) :: var_j
    integer :: var_k

!$omp target map(tofrom: var_j%var_g%var_b)
    do var_k = 1, 10
      var_j%var_g%var_b(var_k) = var_k * 2
    end do
!$omp end target

  print *, var_j%var_g%var_b

  do var_k = 1, 10
    if (var_j%var_g%var_b(var_k) /= var_k * 2) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print*, "======= FORTRAN test passed! ======="
end program prog_a
