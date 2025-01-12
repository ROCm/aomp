  Program test31b_reduction
  implicit none

  integer, parameter :: N=1000
  integer :: i
  double precision,dimension(:),allocatable :: toto
  double precision :: tata

  allocate(toto(2))
  toto=0._8
  tata=0._8  

  !$OMP TARGET map(toto,tata)
  !$OMP TEAMS reduction(+: toto) reduction(max: tata)
  !$OMP DISTRIBUTE PARALLEL DO reduction(+: toto) reduction(max: tata)
  do i = 1, N
     toto(1)=toto(1)+1._8
     if (tata<abs(1._8/(128.2_8-real(i,8)))) tata=abs(1._8/(128.2_8-real(i,8)))
  end do
  !$OMP END DISTRIBUTE PARALLEL DO
  !$OMP END TEAMS
  !$OMP END TARGET
  
  print*,'toto(1) = ',toto(1),', correct answer is: toto(1) = ',N
  print*,'tata = ',tata, ', correct answer is: tata = ',abs(1._8/(128.2_8-real(128,8)))
  deallocate(toto)

  end program test31b_reduction
