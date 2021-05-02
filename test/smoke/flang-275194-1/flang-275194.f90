program test
integer :: i,k
real,pointer :: x(:)
logical ::FLAG
allocate(x(10))
k=0
FLAG=TRUE

!$omp target parallel do if(target:FLAG)
do i=1, 10
x(i)=1
enddo
!$omp end target parallel do

end program test
