! Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
! SPDX-License-Identifier:  MIT

module mod_a
    use, intrinsic :: ISO_Fortran_env
    implicit none

real(kind=REAL64), allocatable :: var_a(:), var_b(:), var_c(:)
integer(kind=4) :: var_d

contains

subroutine sub_a(var_e)
    implicit none
    integer(kind=4) :: var_e
    var_d = var_e
    allocate( var_a(1:var_d), var_b(1:var_d), var_c(1:var_d) )
    !$omp target enter data map(alloc: var_a,var_b,var_c)
end subroutine sub_a

subroutine sub_b()
    implicit none
    !$omp target exit data map(delete: var_a,var_b,var_c)
    deallocate( var_a, var_b, var_c)
end subroutine sub_b

subroutine sub_c()
    implicit none 
!$omp target exit data map(from: var_a,var_b,var_c)
end subroutine sub_c

subroutine sub_d(var_e)
    implicit none
    integer(kind=4) :: var_e
    integer :: var_f
    var_d = var_e
!$omp target
    do var_f = 1, var_d
        var_a(var_f) = 0
        var_b(var_f) = 2
        var_c(var_f) = 3
    end do
!$omp end target
end subroutine sub_d

subroutine sub_e(var_e)
    implicit none
    integer(kind=4) :: var_e
    integer :: var_f
    var_d = var_e
!$omp target parallel do
do var_f = 1, var_d
    var_a(var_f) = var_b(var_f) + var_c(var_f)
end do

!$omp target teams distribute parallel do
    do var_f = 1, var_d
        var_a(var_f) = var_a(var_f) + var_b(var_f) + var_c(var_f)
    end do
end subroutine sub_e

subroutine sub_f(var_e)
    implicit none
    integer(kind=4) :: var_e
    integer :: var_f
    var_d = var_e

    do var_f = 1, var_d
        print *, var_a(var_f)
        print *, var_b(var_f)
        print *, var_c(var_f)
    end do
end subroutine sub_f

end module mod_a

program prog_a
    use mod_a
    implicit none

    call sub_a(10)
    call sub_d(10)
    call sub_e(10)
    call sub_c()
    call sub_f(10)
    call sub_b()
end program
