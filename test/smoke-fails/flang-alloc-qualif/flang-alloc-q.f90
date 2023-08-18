program allocQ
implicit none
  integer, parameter :: n = 100
  real, allocatable , device , dimension(:,:) :: arr
  real, allocatable , device , dimension(:) :: map
  integer :: i,k

  allocate(arr(0:1,n/10), map(n))
  !$omp target enter data map(alloc:arr,map)
  !$omp target teams distribute parallel do simd
  do i = 1, n
    map(i) = i
  enddo
end program allocQ
