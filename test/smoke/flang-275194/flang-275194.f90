program test
      integer :: i,k
      real,pointer :: x(:)
      allocate(x(10))
      k=0

      !$omp target parallel do if(k)
      do i=1, 10
         x(i)=1
      enddo
      !$omp end target parallel do

end program test
