! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    INTEGER :: var_a(10) = (/0,0,0,0,0,0,0,0,0,0/)
    integer :: var_b, var_c

    !$omp target map(tofrom:var_a(1:10))
        do var_b = 1, 10
            var_a(var_b) = var_b
        end do
    !$omp end target

    do var_c = 1, 10
        if (var_a(var_c) /= var_c) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    print*, "======= FORTRAN Test passed! ======="
end program
