subroutine _compute_dev(sum_dev, a_dev, b_dev, c_dev)
    integer i,j,k
    REAL(8), dimension(100,100,100), intent(in) :: b_dev, c_dev
    REAL(8), dimension(100,100,100), intent(out) :: a_dev
    REAL(8), intent(inout) :: sum_dev
    integer :: nsize = 100

!$omp declare target
#define PDO
#ifdef PDO
!$omp parallel do
#endif
    do i=1,nsize
        do j=1,nsize
            do k=1,nsize
                a_dev(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
            end do
        end do
    end do
end subroutine _compute_dev

program main
    REAL(8) :: sum_dev = 0
    REAL(8), dimension(100,100,100) :: a_dev, b_dev, c_dev, E_dev
    integer :: nsize = 100
!$omp declare target(_compute_dev)
    do i=1,nsize
        do j=1,nsize
            do k=1,nsize
                b_dev(i,j,k) = 1
                c_dev(i,j,k) = 2
            end do
        end do
    end do

!$omp target update to(b_dev,c_dev,sum_dev,nsize)
!$omp target
    CALL _compute_dev(sum_dev, a_dev, b_dev, c_dev)
!$omp end target
!$omp target update from(a_dev,sum_dev)

    print *,nsize
    do i=1,nsize
        do j=1,nsize
            do k=1,nsize
                E_dev(i,j,k) = b_dev(i,j,k) * c_dev(i,j,k) * i * nsize*nsize + j * nsize + k
            end do
        end do
    end do

    do i=1,3
        do j=1,3
            do k=1,3
                print *, a_dev(i,j,k)
                if (a_dev(i,j,k) .ne. E_dev(i,j,k)) then
                  print *, "Failed", E_dev(i,j,k)
                  stop 2
                endif
            end do
        end do
    end do
end program main
