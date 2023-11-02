program target_nothing
    implicit none

    integer, parameter :: N = 16
    double precision, dimension(:), allocatable :: arr
    integer :: i

    allocate(arr(N))

    arr(:) = 10.0d0

!$omp nothing

    do i = 1,N
        arr(i) = 42.0d0
    end do
    deallocate(arr(N))
    print *, "PASS"
    return
end program
