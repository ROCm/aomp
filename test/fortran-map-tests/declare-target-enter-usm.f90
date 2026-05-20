! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
    integer :: var_a = 10
    integer :: var_b(10)
    integer, allocatable :: var_c
    integer, allocatable :: var_d(:)
!$omp requires unified_shared_memory
!$omp declare target(var_a)
!$omp declare target(var_b)
!$omp declare target(var_c)
!$omp declare target(var_d)
 contains
end module
program prog_a
 use mod_a
 implicit none
 integer :: var_e

 allocate(var_c)
 allocate(var_d(10))

 var_c = 25

!$omp target
    var_a = var_a + 10
!$omp end target

!$omp target
  do var_e = 1, 10
    var_b(var_e) = var_a
  enddo
!$omp end target

!$omp target
    var_c = var_c + 25
!$omp end target

!$omp target
  do var_e = 1, 10
    var_d(var_e) = var_c
  enddo
!$omp end target

  print *, var_a
  print *, var_b
  print *, var_c
  print *, var_d

  if (var_a /= 20) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_a = 1, 10
    if (var_b(var_a) /= 20) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  if (var_c /= 50) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
  end if

  do var_a = 1, 10
    if (var_d(var_a) /= 50) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if
  end do

  print *, "======= FORTRAN Test Passed! ======="
end program
