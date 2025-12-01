program main
  implicit none

  real, parameter :: num_iters = 1000000
  integer, parameter :: arr_size = 10
  real :: arr(arr_size)
  real :: expected_arr(arr_size)
  integer :: i

  do i = 1, arr_size
    arr(i) = i
    expected_arr(i) = i + num_iters
  end do

  !$omp target map(tofrom: arr)
  call foo(arr)
  !$omp end target

  print *, "result: ", arr

  if (any(arr /= expected_arr)) then
    print *, "Incorrect result! (actual): ", arr, " vs. (expected): ", expected_arr
    stop 1
  end if

contains
  subroutine foo(arr)
    implicit none
    integer :: i
    real, intent(inout) :: arr(arr_size)

    !$omp parallel do reduction(+: arr)
    do i = 1, num_iters
      arr = arr + 1
    end do
  end subroutine
end program
