! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none
    type :: dtype_a
        integer                         :: var_a
    end type dtype_a

    type :: dtype_b
        integer,      pointer, contiguous :: var_b(:) => null()
        real(8),      pointer, contiguous :: var_c(:,:) => null()
        integer :: var_d
        real(8),      pointer, contiguous :: var_e(:,:,:) => null()
    end type dtype_b

    type :: dtype_c
        type(dtype_b),      pointer     :: var_f(:) => null()
        type(dtype_a),     pointer     :: var_g(:)  => null()
        integer :: var_h
    end type dtype_c

    integer :: var_i, var_j
    type(dtype_c), pointer :: var_k => null()

   var_j = 4
   allocate(var_k)
   allocate(var_k%var_f(var_j))
   allocate(var_k%var_g(var_j))
   loop_a: do var_i = 1, var_j
      allocate(var_k%var_f(var_i)%var_e(2,2,2))
      allocate(var_k%var_f(var_i)%var_b(2))
      allocate(var_k%var_f(var_i)%var_c(10,10))
   enddo loop_a


    !$omp target enter data map(to:var_k)
    !$omp target enter data map(always,to:var_k%var_f)
    !$omp target enter data map(always,to:var_k%var_g)
   do var_i = 1, var_j
    !$omp target enter data map(always,to:var_k% var_f(var_i)% var_c)
    !$omp target enter data map(always,to:var_k% var_f(var_i)% var_b)
    !$omp target enter data map(always,to:var_k% var_f(var_i)% var_e)
   end do

    do var_i = 1, var_j
        if (associated(var_k%var_f(var_i)%var_c)) then
            print *, "var_c IS ASSOCIATED 1"
       endif
    enddo

    loop_b: do var_i = 1, var_j
!$omp target update from(var_k% var_f(var_i)% var_d)
!!$omp target update from(var_k% var_f(var_i)% var_c)
!$omp target update from(var_k% var_f(var_i)% var_e)
    deallocate(var_k%var_f(var_i)%var_c)
    deallocate(var_k%var_f(var_i)%var_e)
    deallocate(var_k%var_f(var_i)%var_b)
    enddo loop_b

    do var_i = 1, var_j
        if (associated(var_k%var_f(var_i)%var_c)) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
       endif
    enddo

    do var_i = 1, var_j
        if (associated(var_k%var_f(var_i)%var_e)) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
       endif
    enddo

    do var_i = 1, var_j
        if (associated(var_k%var_f(var_i)%var_b)) then
            print *, "======= FORTRAN Test Failed! ======="
            stop 1
       endif
    enddo

    print *, "======= FORTRAN Test Passed! ======="
end program prog_a
