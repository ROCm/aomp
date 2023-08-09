subroutine print_test(x)
  real, intent(in), dimension(10) :: x
  integer i
  do i = 1, 10
    PRINT *, x(i)
  end do
end subroutine
        
program main
  REAL :: sp(10) = (/0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5/)

!$omp target map(tofrom:sp)
  do i = 1, 10
    sp(i) = sp(i) + i
  end do
!$omp end target

call print_test(sp)

do i = 1, 10
  if (sp(i) /= i + 0.5) then
    print*, "======= FORTRAN Test Failed! ======="
    stop 1    
  end if  
end do 

print*, "======= FORTRAN Test passed! ======="

end program 
