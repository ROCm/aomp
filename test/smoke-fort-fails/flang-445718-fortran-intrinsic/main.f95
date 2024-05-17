!Remark: Test passes if we use real(4) instead of real(8)
program test
implicit none
    real(8) :: x
    integer :: i
    real(8) :: t
    x = 1D0
    t = 2D0
!$omp target  map(from: x) map(to:t)
     x = exp( t)
!$omp end target 
    if ( x .ne. exp(t)) then
      stop 1
    endif
    print *, "Test passed"
end program test
