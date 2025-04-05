program factorial
    implicit none
    integer(kind=16), parameter :: N = 33
    integer(kind=16) :: i, calls, total
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
            else
                f = n * fact(n-1)
            end if
        end function fact
end program factorial
