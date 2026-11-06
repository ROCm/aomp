program p
  integer :: i
  double precision :: a(10)
  !$omp target teams distribute parallel do nowait
  do i = 1, 10
    a(i) = 1d0
  end do
  !$omp taskwait
end program
