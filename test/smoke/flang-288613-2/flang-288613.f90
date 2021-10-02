module test_aomp

implicit none

integer, parameter :: rstd = 8
integer :: nsize
REAL(rstd), allocatable :: a_dev(:,:,:), b_dev(:,:,:), c_dev(:,:,:)
integer :: def_async_q = 0

contains
    subroutine dec_val_dev()

    end subroutine dec_val_dev

    subroutine _compute_dev()
        integer i,j,k,l,m
        real(rstd), dimension(nsize,nsize,nsize) :: tmp

!$omp target teams
        do i=1,nsize
#define PRIVATE_ARRAY 1
#ifdef PRIVATE_ARRAY
!$omp parallel do simd private(tmp)
#else
!$omp parallel do simd
#endif
            do j=1,nsize
                do k=1,nsize
                    tmp(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
                    if (tmp(i,j,k) > 10) then
                        a_dev(i,j,k) = tmp(i,j,k) 
                    else
                        a_dev(i,j,k) = 0
                    end if
                end do
            end do
        end do
!$omp end target teams
    end subroutine _compute_dev

    subroutine compute_dev()
!$omp target update to(b_dev,c_dev,nsize)
        CALL _compute_dev()
!$omp target update from(a_dev)
    end subroutine compute_dev

    subroutine init(n_size)
        integer n_size
        integer i,j,k

        nsize = n_size
        allocate(a_dev(nsize,nsize,nsize), b_dev(nsize,nsize,nsize), c_dev(nsize,nsize,nsize))
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

end module test_aomp

program test
   use test_aomp
   call init(100)
   call compute_dev()
   call deinit()

end program test
