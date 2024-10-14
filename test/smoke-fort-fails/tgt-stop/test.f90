program main
    implicit none
!$omp target
    stop 2
!$omp end target
    print *, "How did I get here?"
end
