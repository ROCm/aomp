! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer,  pointer :: var_a(:)
    integer, target :: var_b(10)
    integer :: var_c

    var_a => var_b

    ! Should have auto attach applied if my reading is
    ! correct and automatically attach to ref_ptee. So
    ! internally we implicitly apply the attach map
    ! type.
    !$omp target enter data map(ref_ptr, to: var_a)
    !$omp target enter data map(ref_ptee, to: var_a)

    ! should in theory memory access fault if we haven't attached
    ! correctly above. But if all went well should go fine.
    !$omp target map(to: var_c)
        do var_c = 1, 10
            var_a(var_c) = var_c
        end do
    !$omp end target

    ! Don't care about the descriptor, but we do want to
    ! deallocate it and only it and then map the data
    ! back. Doing it in a weird-ish order to test we can
    ! delete the descriptor separately and still pull the
    ! data back.
    !$omp target exit data map(ref_ptr, delete: var_a)
    !$omp target exit data map(ref_ptee, from: var_a)

    print *, var_a

    do var_c = 1, 10
        if (var_a(var_c) /= var_c) then
            print*, "======= FORTRAN Test Failed! ======="
            stop 1
        endif
    end do

    print*, "======= FORTRAN Test Passed! ======="
end program
