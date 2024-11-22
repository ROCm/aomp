program test
implicit none
    real(8) :: x
    integer :: i
    real(8) :: t
    x = 1D0
    t = 2D0
!$omp target  map(from: x) map(to:t)
     x = sign(t, -2D0)
!$omp end target
    if ( x .ne. sign(t, -2D0)) then
      stop 1
    endif
    print *, "Test passed"
end program test
