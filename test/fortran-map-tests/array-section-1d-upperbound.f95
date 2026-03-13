! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    INTEGER :: var_a(10) = (/1,2,3,4,5,6,7,8,9,10/)
    INTEGER :: var_b(10) = (/0,0,0,0,0,0,0,0,0,0/)
    integer :: var_c

    !$omp target map(tofrom:var_a(2:6)) map(tofrom:var_b(2:6))
        do var_c = 2, 10
            var_b(var_c) = var_a(var_c)
        end do
    !$omp end target

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

    print*, "======= FORTRAN Test passed! ======="
end program
