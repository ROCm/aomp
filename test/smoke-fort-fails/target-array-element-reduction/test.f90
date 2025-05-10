program main
  implicit none
  integer, parameter :: N = 2
  integer :: v(N)
  v(:) = 0

  !$omp target parallel map(tofrom: v(1:N)) num_threads(2) reduction(+:v(1))
    v(1) = v(1) + 1
  !$omp end target parallel

  !$omp target map(tofrom: v(1:N))
    !$omp parallel num_threads(2) reduction(+:v(2))
      v(2) = v(2) + 1
    !$omp end parallel
  !$omp end target

  if (v(1) .ne. 2 .or. v(2) .ne. 2) then
    stop 1
  end if
end program
