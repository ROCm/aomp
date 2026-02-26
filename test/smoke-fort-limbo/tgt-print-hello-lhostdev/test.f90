program main
    implicit none
!$omp target
    print *, "Hello OpenMP"
    print *, "Hello World"
!$omp end target
end
