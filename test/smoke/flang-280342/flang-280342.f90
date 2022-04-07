
module unit_test_0

implicit none

integer, parameter :: rstd = 8
integer :: nsize
integer :: sum_dev
REAL(rstd), allocatable :: a_dev(:,:,:), b_dev(:,:,:), c_dev(:,:,:)
integer :: def_async_q = 0
!$omp declare target(a_dev,b_dev,c_dev,sum_dev,nsize)

contains
    subroutine _compute_dev()
        integer i,j,k
#ifdef TEST
!$omp declare target
#endif
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    a_dev(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
                    sum_dev = sum_dev + b_dev(i,j,k) * c_dev(i,j,k)
                end do
            end do
        end do

    end subroutine _compute_dev

    subroutine compute_dev()

#ifdef TEST
!$omp declare target(_compute_dev)
#endif

!$omp target update to(b_dev,c_dev,sum_dev,nsize)

#ifdef TEST
!$omp target
#endif
        CALL _compute_dev()
#ifdef TEST
!$omp end target
#endif

!$omp target update from(a_dev,sum_dev)
    end subroutine compute_dev

    subroutine init(n_size)
        integer n_size
        integer i,j,k

        nsize = n_size
        allocate(a_dev(nsize,nsize,nsize), b_dev(nsize,nsize,nsize), c_dev(nsize,nsize,nsize))
        sum_dev = 0
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    b_dev(i,j,k) = 1
                    c_dev(i,j,k) = 2
                end do
            end do
        end do
    end subroutine init

    subroutine deinit()

        deallocate(a_dev, b_dev, c_dev)
    end subroutine deinit

end module unit_test_0

program test
    use unit_test_0

    call init(100)
    call compute_dev

    call deinit()
end program test
