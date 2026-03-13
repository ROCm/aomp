! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer,  pointer :: var_a(:), var_b(:)
    integer, target :: var_c(10), var_d(10)
    integer :: var_e

    var_a => var_d
    var_b => var_c

    do var_e = 1, 10
        var_a(var_e) = var_e
        var_b(var_e) = 0
    end do

    !$omp target map(tofrom:var_a(2:6)) map(tofrom:var_b(2:6))
        do var_e = 1, 10
            var_b(var_e) = var_a(var_e)
        end do
    !$omp end target

    do var_e = 1, 10
        print *, var_b(var_e)
    end do

    if (var_b(1) /= 0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    do var_e = 2, 6
        if (var_b(var_e) /= var_e) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    do var_e = 7, 10
        if (var_b(var_e) /= 0) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    print*, "======= FORTRAN Test passed! ======="
end program
