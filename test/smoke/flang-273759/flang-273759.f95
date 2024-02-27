! example of simple Fortran AMD GPU offloading
program main
  parameter (nsize=1000)
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
! CHECK: DEVID
  write(6,*)"Success: if a diagnostic line starting with DEVID was output"
  ! Always return 1 to show failure in check_smoke_fails.sh. Once
  ! the depend clause is supported on map this can be reverted.
! call exit(1)
end
subroutine foo(a,b,c)
  parameter (nsize=1000)
  real a(nsize), b(nsize), c(nsize)
  integer i
!$omp target map(from:a) map(to:b,c) depend(out:omp_q)
!$omp parallel do
  do i=1,nsize
    a(i) = b(i) * c(i) + i
  end do
!$omp end target
  return
end
