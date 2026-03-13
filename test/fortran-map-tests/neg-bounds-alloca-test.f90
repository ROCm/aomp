! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type dtype_a
        real(4), pointer :: var_a(:,:,:) => null()
    end type
    type(dtype_a), allocatable :: var_b(:,:)
    integer :: var_c

    allocate(var_b(-1:1,1:3))
    do var_c = 1, 3
        allocate(var_b(-1,var_c)%var_a(3,3,3))
    enddo

    !$omp target map(tofrom: var_b(-1,1)%var_a)
        var_b(-1,1)%var_a(2,2,2) = 30
    !$omp end target

    if (var_b(-1,1)%var_a(2,2,2) /= 30) then
        print*, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    print*, "======= FORTRAN test passed! ======="
end program prog_a
