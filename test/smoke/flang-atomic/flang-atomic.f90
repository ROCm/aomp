program test
      integer  :: i,j,k,m
      real     :: R(100), IM(100)
      complex  :: B(100)

      do i=1, 10
      B(i)=0
      enddo

!$omp target teams distribute map(tofrom:B)
      do i=1, 2
!$omp atomic update
      B(2)=B(2)+1
      enddo
!$omp end target teams distribute

      write(*,*) "B= ", B(2)

      end program test
