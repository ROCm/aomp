program loop_reduction_repro
  implicit none
  integer :: i, intarr(2), exp1, exp2, errcnt
  integer, parameter :: n = 1000, s1 = 1, s2 = 2
  intarr(1) = s1
  intarr(2) = s2

  !$omp target teams loop reduction(+:intarr)
  do i=1,n
    intarr(1) = intarr(1) + 1
    intarr(2) = intarr(2) + i
  end do
  write(*,*) n, intarr

  exp1 = n + s1
  exp2 = (n*(n+1))/2 + s2
  write(*,*) n, exp1, exp2

  errcnt = 0
  if (intarr(1) .ne. exp1) then
    print *, "FAILED chk1: ", intarr(1), " != ", exp1
    errcnt = errcnt + 1
  endif
  if (intarr(2) .ne. exp2) then
    print *, "FAILED chk2: ", intarr(2), " != ", exp2
    errcnt = errcnt + 1
  endif
  if (errcnt .gt. 0) then
    stop errcnt
  endif
  print *, "PASSED"
end program
