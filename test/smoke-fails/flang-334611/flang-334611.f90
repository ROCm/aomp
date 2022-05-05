! example of simple Fortran AMD GPU offloading
program main
  parameter (nsize=1000000)
  real a(nsize), b(nsize), c(nsize)
  integer i

  do i=1,nsize
    a(i)=0
    b(i) = i
    c(i) = 10
  end do

  call foo(a,b,c)

  write(6,*)"a(1)=", a(1), "    a(2)=", a(2)
  if (a(1).ne.11 .or. a(2).ne.22) then
    write(6,*)"ERROR: wrong answers"
    stop 2
  endif
  write(6,*)"Success: results compared ok"
  return
end
subroutine foo(a,b,c)
  parameter (nsize=1000000)
  real a(nsize), b(nsize), c(nsize)
  integer i
!$omp target data map(from:a) map(to:b,c)
  do j=1,200
!$omp target teams distribute parallel do private(i) nowait
      do i=1,nsize
        a(i) = b(i) * c(i) + i
      end do
  enddo
!$omp end target data
  return
end
