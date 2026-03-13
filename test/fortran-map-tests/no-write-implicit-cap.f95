! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    REAL :: var_a = 0.5
    REAL :: var_b = 0.5
    !$omp target map(tofrom:var_b)
        var_b = var_a + var_b
        var_a = 15
    !$omp end target

PRINT *, var_a
PRINT *, var_b


if (var_a /= 0.5 .OR. var_b /= 1.0) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1
end if

print*, "======= FORTRAN Test passed! ======="

end program
