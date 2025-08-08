program test
      integer :: i
      integer,parameter :: N = 10 
      complex(4) :: A(N), B(N)

      do i = 1, N
        A(i) = i * (2, 1)
        B(i) = A(i) ** 2
      enddo

!$omp target  parallel do map(tofrom:A)
      do i = 1, N
        A(i) = A(i) ** 2
      enddo
!$omp end target parallel do

      write(*, *), "A=", A
      write(*, *), "B=", B

      do i = 1, N
        if (A(i) /= B(i)) then
          write(*,*) 'Error A != B', A(i), B(i)
          stop 1
        endif
      enddo
end program test
