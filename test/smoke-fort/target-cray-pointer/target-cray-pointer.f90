! Test Cray Pointers

! Test Scalar Case

subroutine cray_scalar()
  integer :: i, pte
  integer :: data = 3
  integer :: j = -3
  pointer(ptr, pte)
  !$omp target map(ptr)
  ptr = loc(data)
  !$omp end target

  i = pte
  print *, i

  pte = j
  print *, data, pte

end

! Test Derived Type Case

subroutine cray_derivedType()
  integer :: pte, k
  type dt
    integer :: i, j
  end type
  type(dt) :: xdt
  pointer(ptr, pte)
  
  xdt = dt(-1, -3)
  !$omp target map(ptr, k, pte)
  ptr = loc(xdt)


  k = pte
  !$omp end target
  print *, k


  pte = k + 2
  print *, xdt, pte

end

! Test Ptr arithmetic Case

subroutine cray_ptrArth()
  integer :: pte, i
  pointer(ptr, pte)
  type dt
    integer :: x, y, z
  end type
  type(dt) :: xdt

  xdt = dt(5, 11, 2)
  !$omp target map(ptr, k, pte, i)
  ptr = loc(xdt)
  ptr = ptr + 4
  i = pte
  print *, i


  ptr = ptr + 4
  pte = -7
  !$omp end target
  print *, xdt

end

! Test Array element Case

subroutine cray_arrayElement()
  integer :: pte, k, data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  !$omp target map(ptr, k, pte, data)
  ptr = loc(data(2))

  k = pte(3)
  print *, k

  pte(2) = -2
  print *, data
  !$omp end target

end

! Test 2d Array element Case

subroutine cray_2darrayElement()
  integer :: pte, k, data(2,4)
  pointer (ptr, pte(2,3))
  data = reshape([1,2,3,4,5,6,7,8], [2,4])
  !$omp target map(ptr, k, pte, data)
  ptr = loc(data(2,2))

  k = pte(1,1)
  print *, k

  pte(1,2) = -2
  print *, data
  !$omp end target

end

! Test Whole Array case

subroutine cray_array()
  integer :: pte, k(3), data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  !$omp target map(ptr, k, pte, data)
  ptr = loc(data(2))

  k = pte
  print *, k

  pte = -2
  print *, data
  !$omp end target

end

! Test Array Section  case

subroutine cray_arraySection()
  integer :: pte, k(2), data(5)
  pointer (ptr, pte(3))
  data = [ 1, 2, 3, 4, 5 ]
  ptr = loc(data(2))

  k = pte(2:3)
  print *, k

  pte(1:2) = -2
  print *, data

end

! Test Cray pointer declared in a module
module mod_cray_ptr
  integer :: pte
  pointer(ptr, pte)
end module

subroutine test_ptr()
  use mod_cray_ptr
  implicit none
  integer :: x
  !$omp target map(ptr, x)
  ptr = loc(x)
  !$omp end target
end

subroutine test1()
  real x(10), res
  pointer (cp, x)
  !$omp target map(cp, x)
  res = x(7)
  !$omp end target
end subroutine test1

subroutine test2(n)
  integer n
  real x
  pointer (cp, x(3:n))
  !$omp target map(cp, x)
  res = x(7)
  !$omp end target
end subroutine test2

subroutine test3(cp, n)
  character(len=11) :: c(n:2*n), res
  pointer (cp, c)
  !$omp target map(cp, c)
  res = c(7)
  !$omp end target
end subroutine test3

subroutine test4(n)
  character(len=n) :: c, res
  pointer (cp, c)
  res = c
end subroutine test4

subroutine test5()
  type t
     sequence
     real r
     integer i
  end type t
  type(t) :: v
  integer res
  pointer (cp, v(3:11))
  !$omp target map(cp)
  res = v(7)%i
  !$omp end target
end subroutine test5

subroutine test7()
  integer :: pte, arr(5)
  pointer(ptr, pte(5))
  !!$omp target map(arr, ptr)
  arr = pte
  !!$omp end target
end subroutine test7

subroutine assumed_size_cray_ptr
  implicit none
  pointer(ivar,var)
  real :: var(*)
  !$omp target map(ivar)
  !$omp end target
end subroutine
