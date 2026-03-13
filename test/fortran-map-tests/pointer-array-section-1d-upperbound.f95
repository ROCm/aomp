! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer,  pointer :: var_a(:), var_b(:)
    integer :: var_c
    allocate(var_a(10))
    allocate(var_b(10))

    do var_c = 1, 10
        var_a(var_c) = var_c
        var_b(var_c) = 0
    end do

    !$omp target map(tofrom:var_a(2:6)) map(tofrom:var_b(2:6))
        do var_c = 1, 10
            var_b(var_c) = var_a(var_c)
        end do
    !$omp end target

    do var_c = 1, 10
        print *, var_b(var_c)
    end do

    if (var_b(1) /= 0) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    do var_c = 2, 6
        if (var_b(var_c) /= var_c) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    do var_c = 7, 10
        if (var_b(var_c) /= 0) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    deallocate(var_a)
    deallocate(var_b)

    print*, "======= FORTRAN Test passed! ======="
end program
