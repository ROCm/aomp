
program main
  common  /dxyz/ var1(2)
  call foo
  print *,var1
end
subroutine foo
  common  /dxyz/ var1(2)
!$omp declare target (/dxyz/)

!$omp target 
  var1(1) = 1 
  var1(2) = 2
!$omp end target
  return
end
