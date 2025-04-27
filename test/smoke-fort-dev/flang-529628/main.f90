program main
  use omp_lib
  use ISO_FORTRAN_ENV, only: REAL64
  implicit none

  integer :: i, n
  real(REAL64), allocatable :: a(:), b(:), c(:), d(:)

  n = 10
  allocate(a(n), b(n), c(n), d(n))

  ! Initialise a
  do i = 1, n
    a(i) = i*1.0_REAL64
    b(i) = 0.0_REAL64
  end do
  d = cos(a)

  !$omp target map(tofrom:n,a,b,c)
  b = 1.0_REAL64  ! Is not working
  c = cos(a)

  !do i = 1, n
  !  b(i) = 1.0_REAL64   ! Is working
  !end do

  !$omp end target
  ! Affichage
  print*, a
  print *, b
  print *, 'Résultat de cos(a):'
  print *, c
  print *, d

end program main
