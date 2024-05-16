program modulo_test
  implicit none
  integer, parameter :: n=1024
  real :: array(n,n)
  integer :: i
  !$omp declare target map(alloc:array)

  !$omp target teams distribute parallel do simd
  do i=1,size(array)
    array(modulo(i+n-1,n)+1,(i+n-1)/n)=i
  enddo

  do i=1,size(array)
    if (array(modulo(i+n-1,n)+1,(i+n-1)/n) .ne. i) then
       print *, "FAILED"
       stop 2
    endif
  enddo
  print *, "SUCCESS"
  return
end program modulo_test
