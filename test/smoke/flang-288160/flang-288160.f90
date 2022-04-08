module test_aomp

implicit none

integer, parameter :: rstd = 8
integer :: nsize
integer :: sum_cpu, sum_dev
REAL(rstd), allocatable :: a_cpu(:,:,:), b_cpu(:,:,:), c_cpu(:,:,:)
REAL(rstd), allocatable :: a_dev(:,:,:), b_dev(:,:,:), c_dev(:,:,:)
integer :: def_async_q = 0

contains
    subroutine dec_val_dev()

    end subroutine dec_val_dev

    subroutine _compute_dev()
        integer i,j,k,l,m


!$omp target
        do i=1,nsize
!$omp parallel do
            do j=1,nsize
                do k=1,nsize
                    a_dev(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
                end do
            end do
#define OFFLOAD_NEXT_LOOP 1
#ifdef OFFLOAD_NEXT_LOOP
!$omp parallel do
#endif
            do l=1,nsize
                do m=1,nsize
                    sum_dev = sum_dev + a_dev(i,l,m)
                end do
            end do
        end do
!$omp end target
    end subroutine _compute_dev

    subroutine compute_dev()


!$omp target update to(b_dev,c_dev,sum_dev,nsize)


        CALL _compute_dev()

!$omp target update from(a_dev,sum_dev)

    end subroutine compute_dev

    subroutine compute_cpu()
        integer i,j,k

        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    a_cpu(i,j,k) = b_cpu(i,j,k) * c_cpu(i,j,k) * i * nsize*nsize + j * nsize + k
                    sum_cpu = sum_cpu + b_cpu(i,j,k) * c_cpu(i,j,k)
                end do
            end do
        end do
    end subroutine compute_cpu

    subroutine init(n_size)
        integer n_size
        integer i,j,k

        nsize = n_size
        allocate(a_cpu(nsize,nsize,nsize), b_cpu(nsize,nsize,nsize), c_cpu(nsize,nsize,nsize))
        allocate(a_dev(nsize,nsize,nsize), b_dev(nsize,nsize,nsize), c_dev(nsize,nsize,nsize))
        sum_cpu = 0
        sum_dev = 0
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    b_dev(i,j,k) = 1
                    b_cpu(i,j,k) = 1
                    c_dev(i,j,k) = 2
                    c_cpu(i,j,k) = 2
                end do
            end do
        end do
    end subroutine init

    subroutine deinit()

        deallocate(a_cpu, b_cpu, c_cpu)
        deallocate(a_dev, b_dev, c_dev)
    end subroutine deinit

    subroutine compare_results()
        integer i,j,k
        double precision :: error
        double precision, parameter :: error_max = 1.0d-10
        integer :: num_error = 0

        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    error = abs(a_dev(i,j,k) - a_cpu(i,j,k))
                    if (error > error_max) then
                        print *,"[Error]: a_dev(", i, ",", j, ",", k, ",", ") : ", a_dev(i,j,k), " <> ", a_cpu(i,j,k)
                        num_error = num_error + 1
                        if (num_error > 10) then
                            exit
                        endif
                    end if
                end do
            end do
        end do
        if (num_error > 0) then
            print *,"TEST FAILED"
            stop 2
        else
            print *,"TEST PASS"
        endif
    end subroutine compare_results

end module test_aomp

program test
   use test_aomp
   call init(100)
   call compute_cpu()
   call compute_dev()
   call compare_results()
   call deinit()

end program test
