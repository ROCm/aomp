! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

PROGRAM prog_a
    integer, allocatable :: var_a
    allocate(var_a)

    var_a = 0
!$omp target map(tofrom:var_a)
   block_a : BLOCK
      do var_b = 1, 10
        BLOCK
           var_a = var_a + var_b
        END BLOCK
     end do
    END BLOCK block_a
!$omp end target

    print *, var_a

    if (var_a /= 55) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    end if

    print *, "======= FORTRAN Test Passed! ======="
END PROGRAM prog_a
