module tests
contains
  subroutine memcpy_int(src, dst, n)
    integer, intent(in) :: src(:)
    integer, intent(inout) :: dst(:)
    integer, intent(in) :: n
    integer :: i

    !$omp target firstprivate(a) map(from: b)
    do i = 1, 10
       dst(i) = src(i)
    end do
    !$omp end target

  end subroutine memcpy_int

  subroutine memcpy_int_custom_lbub(a0, a1, lb, ub)
    integer, intent(in) :: lb, ub
    integer, intent(in) :: a0(lb:ub)
    integer, intent(inout) :: a1(lb:ub)
    integer :: i
    !$omp target firstprivate(a0) map(tofrom:a1)
    do i = lb, ub
       a1(i) = a0(i)
    end do
    !$omp end target
  end subroutine memcpy_int_custom_lbub
end module tests

program test_firstprivate
  use tests
  implicit none

  logical :: main_result
  
  main_result = .TRUE.
  main_result = test_int_non_allocatable()
  main_result = test_int_allocatable(10)
  main_result = test_int_allocatable_with_bounds(-5, 5)
  if (.not. main_result) then
     print *, "(test_firstprivate): FAIL"
     stop 1
  end if
  print *, "(test_firstprivate): PASS"
contains
  function test_int_non_allocatable() result(test_result)
    integer :: a(10), b(10)
    integer :: i
    logical :: test_result
    call initialize(a, 1, 10)
    call initialize(b, 1, 10, val=0)
    
   call memcpy_int(a, b, 10)
   test_result = match(a, b, 1, 10)
  end function test_int_non_allocatable

  function test_int_allocatable(n) result(test_result)
    integer, allocatable :: a(:), b(:)
    integer, intent(in) :: n
    integer :: i
    logical :: test_result
    
    allocate(a(n))
    allocate(b(n))

    call initialize(a, 1, n)
    call initialize(b, 1, n, val=0)
    
    call memcpy_int(a, b, n)
    test_result = match(a, b, 1, n)

    deallocate(a)
    deallocate(b)
  end function test_int_allocatable

  function test_int_allocatable_with_bounds(lb, ub) result(test_result)
    integer, intent(in) :: lb, ub
    integer, allocatable :: a(:), b(:)
    integer :: i
    logical :: test_result

    allocate(a(lb:ub))
    allocate(b(lb:ub))

    call initialize(a, lb, ub)
    call initialize(b, lb, ub, 0)
    
    call memcpy_int_custom_lbub(a, b, lb, ub)
    test_result = match(a, b, lb, ub)

    deallocate(a)
    deallocate(b)
  end function test_int_allocatable_with_bounds

  subroutine initialize(arr, lb, ub, val)
    integer, optional, intent(in) :: val
    integer, intent(in) :: lb, ub
    integer, dimension(lb:ub), intent(out) :: arr
    integer :: i

    if (.not. present(val)) then
      do i = lb, ub
         arr(i) = i
       end do
    else
       do i = lb, ub
          arr(i) = val
       end do
    end if
  end subroutine initialize

  function match(a, b, lb, ub) result(check_result)
    integer, intent(in) :: lb, ub
    integer, dimension(lb:ub), intent(in) :: a, b
    integer :: i
    logical :: check_result

    do i = lb, ub
       if (a(i) /= b(i)) then
          check_result = .FALSE.
          return
       end if
    end do
    check_result = .TRUE.
    end function match
end program test_firstprivate
