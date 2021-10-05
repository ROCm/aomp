
module unit_test_52

implicit none

integer, parameter :: rstd = 8
integer :: nsize
integer :: sum_cpu, sum_dev
REAL(rstd), allocatable :: a_cpu(:,:,:), b_cpu(:,:,:), c_cpu(:,:,:)
REAL(rstd), allocatable :: a_dev(:,:,:), b_dev(:,:,:), c_dev(:,:,:)
!$acc declare create(a_dev,b_dev,c_dev,sum_dev)
!$omp declare target(a_dev,b_dev,c_dev,sum_dev)

contains
    subroutine dec_val_dev()

    end subroutine dec_val_dev

    subroutine _compute_dev()
        integer i,j,k

!$acc parallel loop gang vector collapse(3) present(a_dev,b_dev,c_dev,sum_dev) reduction(+:sum_dev)
!$omp target teams distribute parallel do simd collapse(3) reduction(+:sum_dev)
!!!$omp target teams distribute simd collapse(3) reduction(+:sum_dev)
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

!$acc update device(b_dev,c_dev,sum_dev)
!$omp target update to(b_dev,c_dev,sum_dev)
        CALL _compute_dev()
!$acc update host(a_dev,sum_dev)
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

        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    error = abs(a_dev(i,j,k) - a_cpu(i,j,k))
                    if (error > error_max) then
                        print *,"[Error]: a_dev(", i, ",", j, ",", k, ",", ") : ", a_dev(i,j,k), " <> ", a_cpu(i,j,k)
                        !!call exit(5566)
                    end if
                end do
            end do
        end do
        error = abs(sum_dev - sum_cpu)
        if (error > error_max) then
            print *, "[Error]: sum : ", sum_dev, " <> ", sum_cpu
            !!call exit(5566)
        end if
    end subroutine compare_results

end module unit_test_52

program test
    use unit_test_52

    call init(100)
    call compute_dev()
    call compute_cpu()
    call compare_results()
    call deinit()
end program test
