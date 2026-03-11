! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    integer :: var_a(10) = (/1,2,3,4,5,6,7,8,9,10/)
    integer :: var_b = 8
    integer :: var_c = 1
    integer :: var_d

!$omp target map(tofrom:var_a(var_c:var_b))
    do var_d = 1, 10
        var_a(var_d) = 42
    end do
!$omp end target

    print *, var_a

    do var_d = 1, var_b
        if (var_a(var_d) /= 42) then
          print *, "======= FORTRAN Test Failed! ======="
          stop 1
        end if
    end do

    do var_d = var_b+1, 10
        if (var_a(var_d) /= var_d) then
          print *, "======= FORTRAN Test Failed! ======="
          stop 1
        end if
    end do

    print*, "======= FORTRAN Test passed! ======="
end program
