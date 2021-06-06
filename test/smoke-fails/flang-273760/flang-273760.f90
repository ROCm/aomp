! example of simple Fortran AMD GPU offloading
program main
  use omp_lib, ONLY : omp_target_is_present
  use iso_c_binding, ONLY: c_loc
  parameter (nsize=1000)
  real a(nsize), b(nsize), c(nsize)
  integer i
  logical cond

  do i=1,nsize
    a(i)=0
    b(i) = i
    c(i) = 10
  end do
  !$omp target enter data map(to:b,c)
  cond = omp_target_is_present(c_loc(b),0) .and. omp_target_is_present(c_loc(c),0)
  write(6,*)"Cond=",cond
  !$omp target teams distribute parallel do map(from:a) if(cond)
    do i=1,nsize
      a(i) = b(i) * c(i) + i
    end do
    !$omp end target teams distribute parallel do
  !$omp target update from(a)
  write(6,*)"a(1)=", a(1), "    a(2)=", a(2)
  if (a(1).ne.11 .or. a(2).ne.22) then
    write(6,*)"ERROR: wrong answers"
    stop 2
  endif
  write(6,*)"Success: if a diagnostic line starting with DEVID was output"
  return
end
