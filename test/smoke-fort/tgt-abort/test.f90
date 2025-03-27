program main
    implicit none
!$omp target
    call abort()
!$omp end target
    print *, "How did I get here?"
end
