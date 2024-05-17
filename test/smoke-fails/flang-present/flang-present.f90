! compile with:
!
!   flang -fopenmp --offload-arch=gfx90a -o target_data_present target_data_present.f90
!
! error:
!
!   F90-S-0034-Syntax error at or near : (target_data_present.f90: 25)
!     0 inform,   0 warnings,   1 severes, 0 fatal for target_data_present
!   F90-S-0034-Syntax error at or near : (target_data_present.f90: 25)
!     0 inform,   0 warnings,   1 severes, 0 fatal for target_data_present
!
program target_data_present
    implicit none

    integer, parameter :: N = 16
    double precision, dimension(:), allocatable :: arr
    integer :: i

    allocate(arr(N))

    arr(:) = 10.0d0

!$omp target data map(tofrom:arr)

!$omp target data map(present,alloc:arr)

!$omp target teams distribute parallel do
    do i = 1,N
        arr(i) = 42.0d0
    end do
!$omp end target teams distribute parallel do

!$omp end target data

!$omp end target data

    write (*,*) 'arr=', arr

    deallocate(arr(N))
end program
