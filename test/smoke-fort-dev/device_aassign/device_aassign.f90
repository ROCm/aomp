!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
module mod
    implicit none
contains
    subroutine assgn(a, b)
        implicit none
        real, dimension(:), allocatable :: a
        real, dimension(:), allocatable :: b

        !$omp target
            a = b + 1.0
        !$omp end target
    end subroutine assgn
end module mod

program bla
    use mod
    implicit none

    integer, parameter :: n = 1024
    real, dimension(:), allocatable :: a
    real, dimension(:), allocatable :: b

    allocate(a(n), b(n))

    b = 41.0

    !$omp target enter data map(to:a, b)
    call assgn(a, b)
    !$omp target exit data map(from: a,b)

    if (sum(a / 42.0) /= real(n)) then
        print '(a)', 'ERROR'
        stop 1
    end if
    print '(a)', 'SUCCESS'

    deallocate(a, b)
end program bla
