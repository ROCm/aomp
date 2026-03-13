! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    integer, allocatable :: var_a(:)
    integer :: var_b
    allocate(var_a(10))

   !$omp target enter data map(alloc: var_a)

    !$omp target
        do var_b = 1, 10
            var_a(var_b) = var_b
        end do
    !$omp end target

    !$omp target exit data map(from: var_a)

    !$omp target exit data map(delete: var_a)

     print *, var_a

     do var_b = 1, 10
        if (var_a(var_b) /= var_b) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    deallocate(var_a)

    print*, "======= FORTRAN Test passed! ======="
end program
