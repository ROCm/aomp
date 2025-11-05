subroutine init_array(ub, v)
  implicit none

  integer, intent(in) :: ub
  real, dimension(1:ub+1, 1:ub+1), intent(out) :: v

  integer :: i, j

  !$omp target teams distribute
  do i=1,6
    !$omp parallel do
    do j=1,6
      v(j, i) = 100.0
    end do
  end do
end subroutine

program main
  implicit none

  real, dimension(:, :), allocatable :: v
  integer :: i, j

  allocate(v(1:6, 1:6))

  !$omp target enter data map(alloc: v)
  call init_array(5, v)
  !$omp target exit data map(from: v)

  do i=1,6
    do j=1,6
      if (v(j, i) /= 100.0) then
        print *, "Expected values: 100.0. Actual values:", v(:,:)
        stop 1
      end if
    end do
  end do

  deallocate(v)
end program
