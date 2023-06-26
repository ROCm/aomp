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
end program