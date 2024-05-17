program test

      implicit none
      integer :: i, j, k, N1, N2
      real:: A(10),B(10),C(100)
      integer :: ret
      N1 = 10
      N2 =100

      do i=1, N1
         A(i)=1
         B(i)=1
      enddo

      do i=1, N2
         C(i)=0
      enddo

      ! fails in collapse clause if non-rect
      !$omp target teams distribute parallel do collapse(2) map(to:A,B) map(tofrom:C)  
      do i=0, N1-1
         do j=1, N1  -i
            C(i*N1+j)=A(j)+B(j)
         enddo
      enddo
      !$omp end target teams distribute parallel do

      do i=0, N1-1
         do j=1, N1  -i
         if ( C(i*N1+j) .ne. A(j)+B(j)) then
            write(*, *), "wrong result", c(i*N1+j), A(j), b(j)
            call exit(1)
         endif
        enddo
      enddo
end program test

