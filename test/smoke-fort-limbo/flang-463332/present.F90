program main
  use omp_lib

  parameter (n=1000)
  integer a(n)

  do i=1,n
     a(i) = 0
  end do

  !$omp target data &
  !$omp& map(present,alloc:a)

  !$omp end target data

end program main
