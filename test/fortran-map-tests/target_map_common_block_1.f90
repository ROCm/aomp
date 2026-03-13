! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    use omp_lib
    integer :: var_a(2), var_b
    common var_b
    var_b = 10
    print *, "var_b before target = ", var_b
    var_a(1) = omp_get_device_num()
    !$omp target map(tofrom:var_a) map(tofrom:var_b)
      var_b = 20
      var_a(2) = omp_get_device_num()
    !$omp end target
    print *, "var_b after target = ", var_b
    print *, "var_a: ", var_a

    if (var_b /= 20) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print *, "======= FORTRAN Test Passed! ======="
end program
