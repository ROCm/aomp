module test_0
    INTEGER :: sp = 1
!$omp declare target link(sp)
end module test_0

program main
use test_0
integer :: new_len 

!$omp target map(tofrom:new_len) map(tofrom:sp)
    new_len = sp
!$omp end target

    PRINT *, new_len
    PRINT *, sp

if (sp /= new_len) then     
    print*, "======= FORTRAN Test Failed! ======="
    stop 1    
end if  

print*, "======= FORTRAN Test passed! ======="

end program