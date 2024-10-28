      function glsc3_acc(a,b,mult,n)
      real a(n),b(n),mult(n)
      real tmp,work(1)
      tmp = 0.0

!$omp target teams distribute parallel do simd map(tofrom:tmp)
!$omp& reduction(+:tmp)
      do 10 i=1,n
         tmp = tmp + a(i)*b(i)*mult(i)
!$omp  simd reduction(+:evx)
        do j=0,n-1
          evx = evx + a(j)
        enddo
 10   continue

      glsc3_acc = tmp
      return
      end
program vmm
    implicit none
    integer, parameter :: N = 100000
    real a(N), b(N), c(N)
    integer i, j, num, flag;
    print *, glsc3_acc(a,b,c,N)
end program


