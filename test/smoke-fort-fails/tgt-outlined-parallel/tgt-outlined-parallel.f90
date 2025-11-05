module mymod
   implicit none
contains
  subroutine reduction_and_val(val, res)
    implicit none
    integer, intent(in) :: val
    integer, intent(inout) :: res
    integer :: i

    !$omp parallel do reduction(+:res)
    do i=1,10
      res=res+val
    end do
  end subroutine

  subroutine assumed_shape_array(arr)
    implicit none
    integer, dimension(:), intent(inout) :: arr
    integer :: i

    !$omp parallel do
    do i=1,10
      arr(i)=arr(i)+2
    end do
  end subroutine

  subroutine reduction_and_temp(val, ib)
    implicit none
    integer, intent(inout) :: val
    integer, intent(in) :: ib
    integer :: i

    !$omp parallel do reduction(+:val)
    do i=1,ib
      val = val + 1
    end do
  end subroutine
end module mymod

program main
  use mymod
  implicit none
  integer :: i, val, a, b(10), c

  val=2
  a=0
  !$omp target teams distribute reduction(+:a)
  do i=1,10
    call reduction_and_val(val, a)
  end do
  if (a /= 200) then
    print *, "Test failed (1): Expected 200 but got", a
    stop 1
  end if

  b=1
  !$omp target map(tofrom:b)
    call assumed_shape_array(b)
  !$omp end target
  do i=1,10
    if (b(i) /= 3) then
      print *, "Test failed (2): Expected all 3 but got", b
      stop 1
    end if
  end do

  c=0
  !$omp target teams distribute reduction(+:c)
  do i=1,10
    call reduction_and_temp(c, 10)
  end do
  if (c /= 100) then
    print *, "Test failed (3): Expected 100 but got", c
    stop 1
  end if
end program
