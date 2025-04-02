program fibnoacci
    implicit none
    integer(kind=8), parameter :: N = 33
    integer(kind=8) :: i, calls, total
    integer(kind=8) :: cache(N)
    total = 0
    do i = 0, N
        calls = 0
        print *, "fib(", i, ") = ", fib(i), "(", calls, "calls )"
        total = total + calls
    enddo
    print *, "total ", total, "calls"

    contains
        recursive function fib(n) result(f)
            implicit none
            integer(kind=8), intent(in) :: n
            integer(kind=8) :: f
            calls = calls + 1
            if (n <= 1) then
                f = n
            else if (cache(n) > 0) then
                f = cache(n)
            else
                f = fib(n-1) + fib(n-2)
                cache(n) = f
            end if
        end function fib
end program fibnoacci
