program test
use omp_lib
integer :: i,k, getpid
real,pointer :: x(:)
logical ::FLAG
allocate(x(10))
k=0
FLAG=.TRUE.

!$omp target parallel do if(target:FLAG)
do i=1, 10
  x(i)=omp_get_thread_num()
enddo
!$omp end target parallel do

print *,x
if (x(2) .eq. 0) then
print *,'failed'
call kill(getpid(), 7)
endif

end program test
