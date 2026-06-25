program minimal
  implicit none
  integer, parameter :: n = 64
  real, allocatable :: b(:,:)
  allocate(b(n,n))
  b = 0.0
  !$OMP target data map(tofrom: b)
  call compute(b)
  !$OMP end target data
  write(*,*) b(1,1)
contains
  subroutine compute(b)
    real, dimension(:,:), intent(inout) :: b
    real, dimension(size(b, 1)) :: tmp   ! automatic array sized from assumed-shape arg
    integer :: i, j, k
    !$OMP target teams distribute parallel do collapse(2) private(tmp)
    do j = 1, size(b, 2)
      do i = 1, size(b, 1)
        do k = 1, size(b, 1)
          tmp(k) = 1.0
        end do
        b(i,j) = tmp(1)
      end do
    end do
  end subroutine
end program
