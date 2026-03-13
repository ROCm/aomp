! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
  contains
subroutine sub_a(var_a)
  implicit none 
  integer :: var_b, var_c
  integer,  allocatable, intent (inout) :: var_a(:)

!$omp target map(tofrom: var_a)
  do var_b = 1, 10
    var_a(var_b) = var_a(var_b) + var_b
  end do
!$omp end target

  print *, var_a

  do var_c = 1, 10
    if (var_a(var_c) /= var_c + var_c) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

end subroutine sub_a
end module

subroutine sub_b
  implicit none
  integer :: var_a, var_b
  integer,  allocatable :: var_c(:)
  allocate(var_c(10))

!$omp target map(tofrom: var_c)
  do var_a = 1, 10
    var_c(var_a) = var_a
  end do
!$omp end target

  print *, var_c

  do var_b = 1, 10
    if (var_c(var_b) /= var_b) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  deallocate(var_c)
end subroutine sub_b

program prog_a
use mod_a
implicit none
integer :: var_a, var_b
integer,  allocatable :: var_c(:)

allocate(var_c(10))

!$omp target map(tofrom: var_c)
  do var_a = 1, 10
    var_c(var_a) = var_a
  end do
!$omp end target

  call sub_b

  print *, var_c

  do var_b = 1, 10
    if (var_c(var_b) /= var_b) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  call sub_a(var_c)

  deallocate(var_c)

  print *, "======= FORTRAN Test Passed! ======="
end program
