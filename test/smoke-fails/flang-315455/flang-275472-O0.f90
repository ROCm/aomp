program test
      integer :: i
      real(8), target :: x(10)
      real(8), pointer :: x_d(:)
      do i=1, 10
         x(i)=1
      enddo
      !$omp target enter data map(to:x)
      !$omp target map(from:x_d)
       x_d => x;
      !$omp end target
      write(*, *), "x_d(1)", x_d(1)
end program test
