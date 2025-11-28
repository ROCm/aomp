program main
  implicit none

  real, allocatable :: scalar_alloc
  integer :: i
  real, parameter :: num_iters = 1000000

  allocate(scalar_alloc)
  scalar_alloc = 0

  !$omp target map(tofrom: scalar_alloc)
  call foo(scalar_alloc)
  !$omp end target

  print *, "result: ", scalar_alloc

  if (scalar_alloc /= num_iters) then
    print *, "Incorrect result! (actual): ", scalar_alloc, " vs. (expected): ", num_iters
    stop 1
  end if

contains
subroutine foo(scalar_alloc)
  implicit none
  integer :: i
  real, allocatable, intent(inout) :: scalar_alloc

  !$omp parallel do reduction(+: scalar_alloc)
  do i = 1, num_iters
    scalar_alloc = scalar_alloc + 1
  end do
end subroutine


end program
