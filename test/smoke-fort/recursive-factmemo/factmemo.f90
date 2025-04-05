program factorial
    implicit none
    integer(kind=16), parameter :: N = 33
    integer(kind=16) :: i, calls, total
    integer(kind=16) :: cache(N)
    total = 0
    do i = 0, N
        calls = 0
        print *, "fact(", i, ") =", fact(i), "(", calls, "calls )"
        total = total + calls
    enddo
    print *, "total ", total, "calls"

    contains
        recursive function fact(n) result(f)
            implicit none
            integer(kind=16), intent(in) :: n
            integer(kind=16) :: f
            calls = calls + 1
            if (n <= 1) then
                f = 1
            else if (cache(n) > 0) then
                f = cache(n)
            else
                f = n * fact(n-1)
                cache(n) = f
            end if
        end function fact
end program factorial
