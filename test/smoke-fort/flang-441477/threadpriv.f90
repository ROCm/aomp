program threadpriv
    implicit none

    ! predetermine data-sharing attribute
    common /vars/ a, b
    integer :: a, b
    !$omp threadprivate(/vars/)

    integer :: c, d

    c = 42
    d = 21

    !$omp parallel default(none) shared(c, d)
        a = 84
        b = 42
        print *, a, b, c, d
    !$omp end parallel
end program threadpriv
