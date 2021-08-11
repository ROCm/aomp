program test
      use omp_lib
      integer :: i,k
      real,pointer :: x(:)
      allocate(x(10))
      k=0

     !$omp target if(k > 0) map(from:x)
      k=2
      !$omp parallel do schedule (static,1)
      do i=1, 10
         x(i)=omp_get_thread_num()
      enddo
      !$omp end parallel do
     !$omp end target
     print *,x
     if (x(1) .eq. 0) print *,"Failed" 
end program test
