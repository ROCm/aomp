! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

program prog_a
    implicit none

    type :: dtype_a
        integer,         pointer, contiguous :: var_a(:) => null()
    end type dtype_a

    type :: dtype_b
        integer                         :: var_b
    end type dtype_b

    type :: dtype_c
        type(dtype_a),      pointer     :: var_c(:) => null()
        type(dtype_b),     pointer     :: var_d(:)  => null()
        integer :: var_e
    end type dtype_c

    type(dtype_c), pointer :: var_f => null()
    integer :: var_g, var_h

   allocate(var_f)
   allocate(var_f%var_c(2))
   allocate(var_f%var_d(2))
   allocate(var_f%var_c(1)%var_a(10))
   allocate(var_f%var_c(2)%var_a(10))

   var_f%var_c(1)%var_a(1) = 20

  !$omp target enter data map(to:var_f)
  !$omp target enter data map(always,to:var_f%var_c)
  !$omp target enter data map(always,to:var_f%var_d)
   do var_g = 1, 2
    !$omp target enter data map(always,to:var_f% var_c(var_g)% var_a)
   end do

   !$omp target map(tofrom: var_h)
        var_h = var_f% var_c(1)% var_a(1)
   !$omp end target

    if (var_h /= var_f%var_c(1)%var_a(1)) then
        print *, "======= FORTRAN Test Failed! ======="
        stop 1
    endif

    print *, var_h
    print *, "======= FORTRAN Test Passed! ======="
end program prog_a
