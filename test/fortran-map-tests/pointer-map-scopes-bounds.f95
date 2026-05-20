! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
  contains
subroutine sub_a(var_a)
  implicit none
  integer,  pointer, intent (inout) :: var_a(:)
  integer :: var_b, var_c

!$omp target map(tofrom: var_a(2:6))
  do var_b = 1, 10
    var_a(var_b) = var_a(var_b) + var_b
  end do
!$omp end target

  print *, var_a

  do var_c = 2, 6
    if (var_a(var_c) /= var_c + var_c) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

end subroutine sub_a
end module

subroutine sub_b
  implicit none
  integer,  pointer :: var_a(:)
  integer :: var_b, var_c
  allocate(var_a(10))

!$omp target map(tofrom: var_a(2:6))
  do var_b = 1, 10
    var_a(var_b) = var_b
  end do
!$omp end target

  print *, var_a

  do var_c = 2, 6
    if (var_a(var_c) /= var_c) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  deallocate(var_a)
end subroutine sub_b


program prog_a
use mod_a
implicit none
integer :: var_b, var_c
integer,  pointer :: var_a(:)

allocate(var_a(10))

!$omp target map(tofrom: var_a(2:6))
  do var_b = 1, 10
    var_a(var_b) = var_b
  end do
!$omp end target

  call sub_b

  print *, var_a

  do var_c = 2, 6
    if (var_a(var_c) /= var_c) then
      print *, "======= FORTRAN Test Failed! ======="
      stop 1
    end if
  end do

  call sub_a(var_a)

  deallocate(var_a)

  print *, "======= FORTRAN Test Passed! ======="
end program
