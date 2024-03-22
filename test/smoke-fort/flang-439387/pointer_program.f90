program test_crayptr
  implicit none
  pointer(ivar,var)
  real*8 var(*) !works with gfortran, but not accepted by flang-new
  !pointer(ivar,var(*)) !alternative equivalent statement pair
  !real*8 var
  real*8, allocatable :: location(:)
  integer i

  allocate(location(10))
  ivar = loc(location)
  do i=1,10
    var(i) = i
  enddo
  write(*,*) location

end program test_crayptr
