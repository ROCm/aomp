program test
      integer :: i,k,N
      real(8) :: T1,T2
      real, dimension(:), allocatable :: x

      k=0
      N=10
      allocate(x(N))

      !$omp target if(target:k)
      !$omp parallel do
      do i=1, N
         x(i)=0
      enddo
      !$omp end target

      T1= omp_get_wtime()
      k = 1
      !$omp target if(target:k)
      !$omp parallel do
      do i=1, N
         x(i)=1
      enddo
      !$omp end target 

      T2=omp_get_wtime()
      write(*,*) "Time= ", (T2-T1)


end program test

