subroutine foo(n, x, y, z)
  integer x(n), y(n), z(n)
!$omp declare target
  do i=1,n
    x(i) = y(i) + z(i)
  end do
  return
end subroutine

program main
  parameter (n=1000)
  integer a(n), b(n), c(n)
  external foo
!$omp declare target (foo)

  do i=1,n
    b(i) = i
    c(i) = i*2
  end do

!$omp target teams map(from:a) map(to:b,c)
  call foo(n, a, b, c)
!$omp end target teams
! should print 3 and 6
  write(6,*) "a(1) =", a(1), "    a(2) =", a(2)
end
