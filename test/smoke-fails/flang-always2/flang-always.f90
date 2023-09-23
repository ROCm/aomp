!
! Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.
!
! x86_64 offloading regression test-suite
!
! Last modified: Apr 2020
!

program foo
  use iso_fortran_env, only: int64
  integer, parameter :: n = 100
  integer(int64) :: arr(n), sum
  integer :: i

  arr = 0
  sum = 0

  !$omp target teams map(always, to: sum) reduction(+:sum)
  !$omp distribute parallel do
  do i = 1, n
    arr(i) = 3
  end do

  !$omp distribute parallel do reduction(+:sum)
  do i = 1, n
    sum = sum + arr(i)
  end do
  !$omp end target teams

  if (sum .ne. 300) then
          print * , "failed"
  else
          print *,"passed"
  endif

end program foo
