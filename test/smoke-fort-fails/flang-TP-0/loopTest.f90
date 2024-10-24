program vmm
    implicit none
    integer, parameter :: N = 100000
    integer a(N), b(N), c(N)
    integer i, j, num, flag;
    num = N
    
!$omp target teams map(to: a,b) map(from: c)
!$omp distribute parallel do
    do j=1,1000
    do i=1,N 
        c(i) = a(i) * b(i)
    end do
    end do
!$omp end target teams
    print *,'done'
end program


