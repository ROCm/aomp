! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer :: var_a(10)
    integer :: var_b

    do var_b = 1, 10
        var_a(var_b) = 100
    end do

    !$omp target enter data map(alloc: var_a)

    !$omp target map(always, tofrom:var_a)
        do var_b = 1, 10
            var_a(var_b) = var_a(var_b) + 10
        end do
    !$omp end target

    do var_b = 1, 10
        print *, var_a(var_b)
    end do

    do var_b = 1, 10
        if (var_a(var_b) /= 110) then
          print *, "======= FORTRAN Test Failed! ======="
          stop 1
        end if
    end do

    print *, "======= FORTRAN Test Passed! ======="
end program
