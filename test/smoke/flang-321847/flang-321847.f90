module test_aomp

implicit none

integer, parameter :: rstd = 8
integer :: nsize
REAL(rstd), allocatable :: a_dev(:,:,:), b_dev(:,:,:), c_dev(:,:,:)
logical :: touch_limit
REAL(rstd) :: limit
integer :: use_gpu = 1, count = 0, count_cpu = 0, count_gpu = 0
!$acc declare create(a_dev,b_dev,c_dev,nsize)
!!$omp declare target(a_dev,b_dev,c_dev,nsize)

contains
    subroutine init()
        integer i,j,k
        nsize = 100

        allocate(a_dev(nsize,nsize,nsize))
        allocate(b_dev(nsize,nsize,nsize))
        allocate(c_dev(nsize,nsize,nsize))
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    b_dev(i,j,k) = 0
                    c_dev(i,j,k) = 0
                end do
            end do
        end do
    end subroutine init

    subroutine print_output()
        integer i,j,k
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    !print *, "a(", i, ",", j, ",", k, ") = ", a_dev(i,j,k)
                end do
            end do
        end do
        print *, "touch limit: ", touch_limit
        print *, "count: ", count
    end subroutine print_output

    subroutine _compute_dev_cpu()
        integer i,j,k
!$omp parallel do reduction(.or.:touch_limit) reduction(+:count)
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    a_dev(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
                    if (a_dev(i,j,k) > limit) then
                        touch_limit = touch_limit .or. .true.
                        count = count + 1
                    else
                        touch_limit = touch_limit .or. .false.
                        count = count + 0
                    end if
                end do
            end do
        end do
    end subroutine _compute_dev_cpu

    subroutine _compute_dev_gpu()
        integer i,j,k
!$omp target teams distribute parallel do reduction(.or.:touch_limit) reduction(+:count) if(target:use_gpu)
        do i=1,nsize
            do j=1,nsize
                do k=1,nsize
                    a_dev(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
                    if (a_dev(i,j,k) > limit) then
                        touch_limit = touch_limit .or. .true.
                        count = count + 1
                    else
                        touch_limit = touch_limit .or. .false.
                        count = count + 0
                    end if
                end do
            end do
        end do
    end subroutine _compute_dev_gpu

    subroutine compute_dev()
!!$omp target update to(b_dev,c_dev,nsize,limit)
        CALL _compute_dev_cpu()
        print *, "count CPU: ", count
        count_cpu = count
        count = 0
        CALL _compute_dev_gpu()
        print *, "count GPU: ", count
        count = 0
        CALL _compute_dev_gpu()
        count_gpu = count
        if (count_cpu == count_gpu) then
            print *, "PASS"
        else
            print *, "FAIL"
            stop 2
        end if
!!$omp target update from(a_dev,count,touch_limit)
    end subroutine compute_dev


end module test_aomp

program test
   use test_aomp

   character(len=12) :: args

   limit= 10099
   touch_limit= .false.

   call get_command_argument(1, args)
   read(args,'(i)') use_gpu
   use_gpu = 1
   call init()
   call compute_dev()
end program test
