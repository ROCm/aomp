program main
  use omp_lib
  implicit none
  integer :: a = 10
  integer :: collect_a = 0
  integer, parameter :: n = 1024

  !$omp target teams map(tofrom: a, collect_a)
  block
  integer :: a = 20
  if (omp_get_team_num() == 0) collect_a = a
  end block
  !$omp end target teams

  if (collect_a == 20) then
     print *,"Success!"
  else
     write(*,*) "collect_a expected 20, now = ", collect_a
     stop 1
  endif

  if (a == 10) then
     print *,"Success!"
  else
     write(*,*) "a expected 10, now = ", a
     stop 1
  endif

end program main
