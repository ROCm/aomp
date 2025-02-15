program test
use omp_lib
integer :: i,k, getpid
real,pointer :: x(:)
logical ::FLAG
allocate(x(10))
k=0
FLAG=.FALSE.


!$omp target parallel do if(FLAG) map(tofrom: x)
do i=1, 10
  x(i)=omp_get_thread_num()
enddo
!$omp end target parallel do
print *,x
if (x(2) .ne. 0) then      
  print *,'failed'
  call kill(getpid(), 7)
endif

end program test
