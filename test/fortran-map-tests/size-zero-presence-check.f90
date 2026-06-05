! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer :: var_a
    integer :: var_b
    real, allocatable :: var_c(:)
    integer :: var_d

    var_a = 10
    var_b = 9

    ! Make it size 0, the main purpose of this test is to make sure
    ! the presence check on a size 0 array returns true and lets us
    ! pass to the end of the program.
    allocate(var_c(var_a:var_b))

    do var_d = var_a, var_b
        var_c(var_d) = real(var_d)
    end do

    !$omp target enter data map(to: var_c)

    !$omp target teams distribute parallel do map(present,alloc: var_c)
    do var_d = var_a, var_b
        var_c(var_d) = var_c(var_d) * 2.0
    end do
    !$omp end target teams distribute parallel do

    !$omp target exit data map(from: var_c)

    do var_d = var_a, var_b
        if (var_c(var_d) /= real(var_d) * 2.0) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
        end if
    end do

    deallocate(var_c)

    print*, "======= FORTRAN Test Passed! ======="
end program
