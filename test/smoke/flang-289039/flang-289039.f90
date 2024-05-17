program test
      integer :: i,j
      real(8) :: A
      real(8), pointer :: B(:)
      allocate(B(100))

      do i=1, 100
      B(i)=0.94063778517935193
      enddo
      !$omp target enter data map(to:B)

      !$omp target enter data map(alloc:A)
      !$omp target
      A=B(1)
      !$omp end target
      !$omp target update from(A)

      write(*,*) "A= ", A, "B(1)= ", B(1)

      !$omp target exit data map(delete:A,B)

      end program test
