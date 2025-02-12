subroutine decrement_val(i, beta)
        !$omp declare target
        real, dimension(*), intent(out) :: beta
        integer, value :: i
        beta(i) = beta(i) - 1
end subroutine
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
  if (a(1).ne.10 .or. a(2).ne.21) then
    write(6,*)"ERROR: wrong answers"
    stop 2
  endif
  write(6,*)"Success: if a diagnostic line starting with DEVID was output"
  return
end
subroutine foo(a,b,c)
  parameter (nsize=1000000)
  real a(nsize), b(nsize), c(nsize)
  integer i
 INTERFACE
    SUBROUTINE decrement_val(i,beta)
        real, dimension(*), intent(out) :: beta
        integer, value :: i
    END SUBROUTINE
  END INTERFACE

!$omp declare target(decrement_val)
!$omp target map(from:a) map(to:b,c)
!$omp parallel do
  do i=1,nsize
    a(i) = b(i) * c(i) + i
    call decrement_val(i,a)
  end do
!$omp end target
  return
end
