program ackermann
    implicit none
    integer(kind=8), parameter :: M = 3, N = 12
    integer(kind=8) :: i, j, calls, total
    integer(kind=8), parameter :: NCACHE = 40000
    integer(kind=8) :: cache(M, NCACHE)
    total = 0
    do i = 0, M
        do j = 0, N
            calls = 0
            print *, "ack(", i, j, ") = ", ack(i, j), "(", calls, "calls )"
            total = total + calls
        enddo
    enddo
    print *, "total ", total, "calls"

    contains
        recursive function ack(m, n) result(f)
            implicit none
            integer(kind=8), intent(in) :: m, n
            integer(kind=8) :: f
            calls = calls + 1
            if (m == 0) then
                f = n + 1
                return
            else if (n == 0) then
                f = ack(m-1, 1_8)
                return
            else if (n <= NCACHE) then
                if (cache(m, n) > 0) then
                    f = cache(m, n)
                    return
                endif
            endif

            f = ack(m-1, ack(m, n-1))
            if (n <= NCACHE) then
                cache(m, n) = f
            endif
        end function ack
end program ackermann
