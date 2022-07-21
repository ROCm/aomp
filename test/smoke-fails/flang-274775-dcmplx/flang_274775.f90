subroutine foo(M)
        double complex :: M
        if (M .ne. (100,100)) then
                write(*,*) "Failed"
                stop 2
        endif
        end subroutine foo

program test
         integer :: i,j
         double complex :: M,N
         double complex,pointer :: A(:)
  
        allocate(A(10))
        M=(0,0)
 
        do i=1, 10
          A(i)=(1,1)
        enddo
 
!$omp target teams distribute map(to:A) map(tofrom:M) reduction(+:M) 
       do i=1, 10
!$omp parallel do reduction(+:M)
         do j=1, 10
           M=M+A(j)
         enddo
!$omp end parallel do
      enddo
!$omp end target teams distribute

       write(*, *) "M=", M
       call foo(M)
end program test
