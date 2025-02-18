module mod
    implicit none
contains
    subroutine routine(a)
        implicit none
        real, dimension(:), optional :: a
        !$omp target data if(present(a)) map(alloc:a)
        !$omp end target data
    end subroutine routine
end module mod
