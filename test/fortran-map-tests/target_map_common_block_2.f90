! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    use omp_lib
    integer :: var_a, var_b
    common var_b
    var_b = 24
    var_a = 12
    print *, "var_b before target = ", var_b
    !$omp target map(tofrom:var_b)
      var_b = var_a
    !$omp end target
    print *, "var_b after target = ", var_b

    if (var_b /= 12) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print *, "======= FORTRAN Test Passed! ======="
end program
