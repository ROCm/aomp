module globals

  real, allocatable :: a(:)
!$omp threadprivate(a)

end module globals

program test

  use omp_lib
  use globals
  implicit none

  integer, parameter :: n=17 ! value is irrelevant
  integer :: i
  real :: sum

  allocate(a(n))
  a(:)=1.0
  sum=0.0

! reduction is irrelevant
!$omp parallel do reduction(+:sum), &
!$omp private(i), &
!$omp copyin(a)
  do i=1,n
     sum=sum+a(i)
     a(i) = 42.0
  enddo
!$omp end parallel do

  print *,sum

end program test
