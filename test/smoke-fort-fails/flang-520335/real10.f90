subroutine routine()
    implicit none
    !$omp declare target
    real(kind=10) :: a
end subroutine routine


program real10
    use, intrinsic :: ieee_arithmetic
    implicit none

    integer, parameter :: rk = 10
    real(kind=rk) :: a
    real(kind=rk) :: b
    real(kind=rk) :: m

    a = 1.0_rk
    b = 1.0_rk

    m = ieee_min(a, b)
    print *, a, b, m

    !$omp target map(tofrom:a,b) map(from:m)
        a = a + 1.0_rk
        b = b - 5.0_rk
        m = ieee_min(a, b)
    !$omp end target
    print *, a, b, m
end program
