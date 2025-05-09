program vadddemo
    use mymod
    implicit none
    integer, parameter :: N = 100000
    real a(N), b(N), c(N), validate(N)
    integer i, num, flag, size;
    num = N
    
    size = sizeof(a(1))
    print *, "type: real"
    print *, "bytes: ", size

    flag = -1
    do i = 1, N
        a(i) = i+1;
        b(i) = i+2;
        validate(i) = a(i) + b(i);
    enddo

    call vadd(a, b, c, N)

    do i = 1, num
        if (c(i) .ne. validate(i)) then
            ! print 1st bad index
            if ( flag .eq. -1 ) then
                print *, "First fail: c(", i, "):", c(i), " != validate(", i, "):", validate(i)
            endif
            flag = i;
        endif
    enddo
    if (flag .eq. -1) then
        print *, "Success"
        call exit(0)
    else
        print *, "Last fail:  c(", flag, "):", c(flag), " != validate(", flag, "):", validate(flag)
        print *, "FAILED"
        call exit(1)
    endif
end program
