! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    type dtype_a
      integer var_a(20)
      integer var_b
    end type dtype_a

    type (dtype_a) var_c(5)
  contains

  subroutine sub_a()
    implicit none
!$omp target map(tofrom: var_c(5))
    var_c(5)%var_a(5) = 500
!$omp end target
  end subroutine

  subroutine sub_b()
    implicit none

!$omp target map(tofrom: var_c(5))
    var_c(5)%var_a(5) = var_c(5)%var_a(5) + 500
!$omp end target
  end subroutine
end module mod_a

program prog_a
   use mod_a

  call sub_a()
  call sub_b()

  print *, var_c(5)%var_a(5)

  if (var_c(5)%var_a(5) /= 1000) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  print*, "======= FORTRAN Test Passed! ======="
end program
