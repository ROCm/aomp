! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    implicit none
    integer, allocatable, dimension(:) :: var_a
    !$omp declare target(var_a)
  end module mod_a

  program prog_a
    use mod_a
    implicit none
    integer :: var_b, var_c

    allocate(var_a(10))

    do var_c = 1, 10
        var_a(var_c) = 0
    end do

!$omp target data map(to:var_a)

  do var_b = 1, 2

    !$omp target
        do var_c = 1, 10
            var_a(var_c) = var_a(var_c) + var_c
        end do
    !$omp end target

! NOTE: Technically doesn't affect the results, but there is a
! regression case that'll cause a runtime crash if this is
! invoked more than once, so this checks for that.
!$omp target update from(var_a)
  end do

!$omp end target data

  print *, var_a

  do var_c = 1, 10
    if (var_a(var_c) /= var_c + var_c) then
        print *, "======= FORTRAN Test Failed! ======="
    end if
  end do

  print *, "======= FORTRAN Test Passed! ======="
END PROGRAM
