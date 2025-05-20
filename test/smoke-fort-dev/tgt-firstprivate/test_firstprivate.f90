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

  subroutine memcpy_char(src, dst)
    character(len=*), intent(in) :: src
    character(len=*), intent(out) :: dst

    !$omp target firstprivate(src) map(from:dst)
    dst = src
    !$omp end target

  end subroutine memcpy_char

end module tests

program test_firstprivate
  use tests
  implicit none

  logical :: main_result
  
  main_result = .TRUE.
  main_result = main_result .AND. test_int_non_allocatable()
  main_result = main_result .AND. test_int_allocatable(10)
  main_result = main_result .AND. test_int_allocatable_with_bounds(-5, 5)
  main_result = main_result .AND. test_char_non_allocatable()
  main_result = main_result .AND. test_char_allocatable()
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
   test_result = match_int(a, b, 1, 10)
   call print_result(test_result, "test_int_non_allocatable")
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
    test_result = match_int(a, b, 1, n)

    deallocate(a)
    deallocate(b)
    call print_result(test_result, "test_int_allocatable")
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
    test_result = match_int(a, b, lb, ub)

    deallocate(a)
    deallocate(b)
    call print_result(test_result, "test_int_allocatable_with_bounds")
  end function test_int_allocatable_with_bounds

  function test_char_non_allocatable() result(test_result)
    character(len=10) :: a, b
    logical :: test_result

    a = "john"

    call memcpy_char(a, b)
    test_result = match_char(a, b)
    call print_result(test_result, "test_char_non_allocatable")
  end function test_char_non_allocatable

  function test_char_allocatable() result(test_result)
    character(len=:), allocatable :: a, b
    integer :: n
    logical :: test_result
    n = 10

    allocate(character(len=n) :: a)
    allocate(character(len=n) :: b)
    a = "john"
    call memcpy_char(a, b)
    test_result = match_char(a, b)

    deallocate(a)
    deallocate(b)
    call print_result(test_result, "test_char_allocatable")

  end function test_char_allocatable

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

  subroutine print_result(res, msg)
    logical, intent(in) :: res
    character(len=*), intent(in) :: msg

    if (res) then
       print *, "PASS: ", msg
    else
       print *, "FAIL: ", msg
    end if
  end subroutine print_result

  function match_int(a, b, lb, ub) result(check_result)
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
  end function match_int

  function match_char(a, b) result(check_result)
    character(len=*), intent(in) :: a
    character(len=*), intent(in) :: b
    logical :: check_result
    if (a .ne. b) then
       check_result = .FALSE.
       return
    end if
    check_result = .TRUE.
  end function match_char
end program test_firstprivate
