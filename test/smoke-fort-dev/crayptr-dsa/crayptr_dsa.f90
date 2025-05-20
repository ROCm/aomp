program test_crayptr
  use omp_lib
  implicit none

  call none_shared()
  call none_private()
  call none_firstprivate()
  call private_shared()
  call private_firstprivate()
  call firstprivate_shared()
  call firstprivate_private()
end program test_crayptr

subroutine none_shared()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(none) shared(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'none_shared', var(1)
  !$omp end parallel
end subroutine

subroutine none_private()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(none) private(ivar) shared(pointee)
    ivar = loc(pointee)
    var(1) = var(1) / 2
    print '(A24,I6)', 'none_private', var(1)
  !$omp end parallel
end subroutine

subroutine none_firstprivate()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(none) firstprivate(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'none_firstprivate', var(1)
  !$omp end parallel
end subroutine

subroutine private_shared()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(private) shared(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'private_shared', var(1)
  !$omp end parallel
end subroutine

subroutine private_firstprivate()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(private) firstprivate(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'private_firstprivate', var(1)
  !$omp end parallel
end subroutine

subroutine firstprivate_shared()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(firstprivate) shared(ivar)
    var(1) = var(1) / 2
    print '(A24,I6)', 'firstprivate_shared', var(1)
  !$omp end parallel
end subroutine

subroutine firstprivate_private()
  implicit none
  integer var(*)
  pointer(ivar,var)
  integer pointee(8)

  pointee(1) = 42
  ivar = loc(pointee)

  !$omp parallel num_threads(1) default(firstprivate) private(ivar) shared(pointee)
    ivar = loc(pointee)
    var(1) = var(1) / 2
    print '(A24,I6)', 'firstprivate_private', var(1)
  !$omp end parallel
end subroutine

