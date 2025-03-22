#ifndef CHKTYPE
#error CHKTYPE not defined
#endif
#ifndef CHKBYTES
#error CHKBYTES not defined
#endif
#define STR(a) _STR(a)
#define _STR(a) #a

subroutine vadd(a, b, c, N)
    implicit none
    CHKTYPE :: a(N), b(N), c(N)
    integer :: N, i

!$omp target map(to: a,b) map(from: c)
!$omp teams distribute parallel do 
    do i=1,N
        c(i) = a(i) + b(i)
    end do
!$omp end target
end subroutine

program vadddemo
    implicit none
    integer, parameter :: N = 100000
    CHKTYPE a(N), b(N), c(N), validate(N)
    integer i, num, flag, size;
    num = N
    
    size = sizeof(a(1))
    print *, "type: ", STR(CHKTYPE)
    print *, "bytes: ", size
    if (size .ne. CHKBYTES) then
        print *, "Error: expected bytes = ", CHKBYTES
        call exit(1)
    endif
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
