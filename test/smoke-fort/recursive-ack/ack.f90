program ackermann
    implicit none
    integer(kind=8), parameter :: M = 3, N = 12
    integer(kind=8) :: i, j, calls, total
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
            else if (n == 0) then
                f = ack(m-1, 1_8)
            else
                f = ack(m-1, ack(m, n-1))
            end if
        end function ack
end program ackermann
