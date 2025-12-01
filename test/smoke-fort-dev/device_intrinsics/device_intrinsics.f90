!
!Copyright © Advanced Micro Devices, Inc., or its affiliates.
!
!SPDX-License-Identifier:  MIT
! 
module mod
    implicit none
contains
    subroutine assgn(a, ma, mi, ms)
        implicit none
        real, dimension(:), allocatable :: a
        real :: ma, mi, ms

        !$omp target
            ma = maxval(a)
            mi = minval(a)
            ms = sum(a)
        !$omp end target
    end subroutine assgn
end module mod

program device_intrinsics
    use mod
    implicit none

    integer, parameter :: n = 1024
    integer :: i
    real, dimension(:), allocatable :: a
    real, dimension(:), allocatable :: b
    real :: ma, mi, ms

    allocate(a(n))
    allocate(b(n))

    a = [( real(i),i=1,n )]
    b = [( real(i),i=1,n )]

    !$omp target enter data map(to:a) map(alloc:ma,mi,ms)
    call assgn(a, ma, mi, ms)
    !$omp target exit data map(delete:a) map(from:ma,mi,ms)

    print *, mi, ma, ms
    if (mi /= minval(b) .or. ma /= maxval(b) .or. ms /= sum(b)) then
        print '(a)', 'ERROR'
        stop 1
    end if
    print '(a)', 'SUCCESS'

    deallocate(a)
end program device_intrinsics
