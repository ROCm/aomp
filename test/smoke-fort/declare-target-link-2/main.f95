module test_0
    implicit none
    INTEGER :: sp = 0
!$omp declare target link(sp)
end module test_0

program main
    use test_0
!$omp target map(tofrom:sp)
    sp = 1
!$omp end target

PRINT *, sp

if (sp /= 1) then     
    print*, "======= FORTRAN Test Failed! ======="
    stop 1    
end if  

print*, "======= FORTRAN Test passed! ======="

end program