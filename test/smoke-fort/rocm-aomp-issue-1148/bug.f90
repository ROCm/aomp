module bug
  implicit none
  contains

  subroutine b(out)
    !$omp declare target
    real, dimension(:), intent(inout) :: out
    out = 13.0
  end subroutine b

  subroutine a(out)
    !$omp declare target
    real, dimension(3) :: arr
    real, dimension(3), intent(inout) :: out
    call b(arr)
    out = arr
  end subroutine a

end module
program use_bug
use bug, only: a
implicit none
real, dimension(3) :: xs
xs = 0
!$omp target map(xs)
call a(xs)
!$omp end target
print *, xs
end program
