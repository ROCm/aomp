program test_crayptr
  implicit none
  real*8 var(*)
  pointer(ivar,var)
  real*8 pointee(8)

  pointee(1) = 42.0
  ivar = loc(pointee)

  !$omp parallel num_threads(2) default(none) shared(ivar)
    print *, var(1)
  !$omp end parallel
end program test_crayptr
