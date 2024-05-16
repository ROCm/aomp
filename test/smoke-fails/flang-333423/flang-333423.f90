program atomic
implicit none
  integer, parameter :: n = 1000
  integer, allocatable :: arr(:,:), map(:)
  integer :: i,k

  allocate(arr(0:1,n/10), map(n))
  !$omp target enter data map(alloc:arr,map)
  !$omp target teams distribute parallel do simd
  do i = 1, n
    map(i) = i
  enddo
  !$omp target teams distribute parallel do simd
  do i=1,n/10
    arr(0,i) = 0.
    arr(1,i) = 0.
  enddo
  !$omp target teams distribute parallel do simd
  do i=1,n
    k = i-n/10*((i-1)/(n/10))
    !$omp atomic update
    arr(0,k) = arr(0,k) + map(n)
    !$omp atomic update
    arr(1,k) = arr(1,k) + map(k)
  enddo
  !$omp target update from(arr)
  !$omp target exit data map(delete:arr,map)
  write(*,*) arr
  
end program atomic
