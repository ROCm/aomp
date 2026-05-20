! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    integer,  pointer :: var_a(:)
    integer, target :: var_b(10)
    integer, target :: var_c(10)
    integer :: var_d, var_e
    logical :: var_f

    var_e = 10
    var_f = .true.

    do var_d = 1, var_e
        var_b(var_d) = 10
        var_c(var_d) = 20
    end do

    var_a => var_b

   ! This should map a,b and map_ptr to device, and attach map_ptr
   ! to a (as it is assigned to it above), and as b is already on
   ! device running through target.
   !$omp target enter data map(ref_ptr_ptee, to: var_a)
   !$omp target enter data map(to: var_c, var_b)

    !$omp target map(to: var_d) map(tofrom: var_f)
        do var_d = 1, var_e
            if (var_a(var_d) /= 10) then
                var_f = .false.
            endif
        end do
    !$omp end target

    var_a => var_c

    ! No attach always to force re-attachment, so we should still
    ! be attached to "a"
    !$omp target map(to: var_d) map(tofrom: var_f)
        do var_d = 1, var_e
            if (var_a(var_d) /= 10) then
                var_f = .false.
            endif
        end do
    !$omp end target

    !$omp target map(to: var_d) map(attach(always): var_a) map(tofrom: var_f)
        do var_d = 1, var_e
            if (var_a(var_d) /= 20) then
                var_f = .false.
            endif
        end do
    !$omp end target

    if (var_f .NEQV. .true.) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    print*, "======= FORTRAN Test Passed! ======="
end program
