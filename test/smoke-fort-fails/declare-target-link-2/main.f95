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

end program