program t270386
! TWO bugs (lack of supported features) in this test case
! testing presence of an  IN_REDUCTION clause on a target directive
! testing presence of both an IF and IN_REDUCTION clause on a target directive
  integer :: sum
  sum = 0
!$omp target parallel in_reduction(+:sum) 
  sum = sum + 1
!$omp end target parallel
!$omp target parallel in_reduction(+:sum) if(sum<99)
  sum = sum + 1
!$omp end target parallel
  print *, "sum = ", sum
  if (sum .ne. 257) then
    print *, "wrong answers"
    stop 2
  endif
end program
 
