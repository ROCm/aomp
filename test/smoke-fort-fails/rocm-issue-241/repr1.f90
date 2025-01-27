program repr1
  integer, dimension(:), allocatable :: xs
  integer :: i, acc
  !$omp declare target (xs)

  allocate(xs(500))
  !$omp target enter data map(alloc:xs)
  xs = 1
  acc = 0
  !$omp target update to(xs)

  !$omp target map(acc)
  do i=1, 500
    acc = acc + xs(i)
  end do
  !$omp end target 
  
  print *, "acc", acc
end program
