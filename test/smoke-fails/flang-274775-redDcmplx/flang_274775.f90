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
 
!$omp target teams distribute parallel do collapse(2) map(to:A) map(tofrom:M) REDUCTION(M) private(j)
       do i=1, 10
         do j=1, 10
           M=M+A(j)
         enddo
      enddo
!$omp end target teams distribute parallel do

       write(*, *) "M=", M
       call foo(M)
end program test
