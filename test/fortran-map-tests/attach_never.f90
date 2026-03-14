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

    var_f = .true.
    var_e = 10

    do var_d = 1, var_e
        var_b(var_d) = 10
        var_c(var_d) = 20
    end do

    var_a => var_b

    ! This should map a and map_ptr to device, and attach map_ptr
    ! to a (as it is assigned to it above).
    !$omp target enter data map(ref_ptr_ptee, to: var_a)

    var_a => var_c

    ! As "b" hasn't been mapped to device yet, the first time it's mapped will
    ! be when map_ptr is re-mapped (implicitly or explicitly), the default behaviour
    ! when LIBOMPTARGET_TREAT_ATTACH_AUTO_AS_ALWAYS is switched off would force attachment
    ! of map_ptr to b as we've assigned it above. To prevent this and test the never
    ! attachment, we can apply attach(never), which prevents this reattachment from occuring
    !$omp target map(to: var_d) map(tofrom: var_f) map(attach(never): var_a)
        do var_d = 1, var_e
            if (var_a(var_d) /= 10) then
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
