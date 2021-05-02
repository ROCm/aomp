program test
         integer :: i,j
         complex :: M,N
         complex,pointer :: A(:)
  
        allocate(A(10))
        M=(0,0)
 
        do i=1, 10
          A(i)=(1,1)
        enddo
 
!$omp target teams distribute map(to:A) map(tofrom:M) reduction(+:M) 
       do i=1, 10
!$omp parallel do
         do j=1, 10
           M=M+A(j)
         enddo
!$omp end parallel do
      enddo
!$omp end target teams distribute

       write(*, *), "M=", M

end program test
