module function_mod
  implicit none
contains
  subroutine takes_integer(my_int)
  !$omp declare target
    integer, intent(inout) :: my_int
    my_int = my_int + 1
  end subroutine takes_integer
end module function_mod

subroutine my_test(my_sum,n)
  use function_mod
  implicit none
  integer, intent(inout) :: my_sum
  integer, intent(in) :: n
  integer :: i

  !$omp target teams distribute parallel do simd reduction(+:my_sum)
  do i=1,n
    call takes_integer(my_sum)
  end do
end subroutine my_test
