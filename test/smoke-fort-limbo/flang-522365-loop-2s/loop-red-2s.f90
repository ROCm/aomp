program loop_reduction_repro
  implicit none
  integer :: i, int1, int2, exp1, exp2, errcnt
  integer, parameter :: n = 1000, s1 = 1, s2 = 2
  int1 = s1
  int2 = s2

  !$omp target teams loop reduction(+:int1,int2)
  do i=1,n
    int1 = int1 + 1
    int2 = int2 + i
  end do
  write(*,*) n, int1, int2

  exp1 = n + s1
  exp2 = (n*(n+1))/2 + s2
  write(*,*) n, exp1, exp2

  errcnt = 0
  if (int1 .ne. exp1) then
    print *, "FAILED chk1: ", int1, " != ", exp1
    errcnt = errcnt + 1
  endif
  if (int2 .ne. exp2) then
    print *, "FAILED chk2: ", int2, " != ", exp2
    errcnt = errcnt + 1
  endif
  if (errcnt .gt. 0) then
    stop errcnt
  endif
  print *, "PASSED"
end program
