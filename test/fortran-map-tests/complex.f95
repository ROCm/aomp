! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    complex  :: var_a
    complex  :: var_b
    var_b = (10,20)

!$omp target map(from:var_a) map(to:var_b)
    var_a = var_b
!$omp end target

    write (*,*) var_a
    write (*,*) var_b

    if (var_b /= (10,20)) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    if (var_a /= (10,20)) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print*, "======= FORTRAN Test passed! ======="
end program prog_a
