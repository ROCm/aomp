module example
contains

  subroutine a()
    integer :: i
    !$omp target teams distribute parallel do
      do i=1,10
          call b(i)
      end do
    !$omp end target teams distribute parallel do
  end subroutine a

end module
