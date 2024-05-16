! Select which VMUL[0123] to use
#define VMUL0

#ifdef VMUL0

subroutine vmul(a, b, c, N)
    implicit none
    integer :: a(N), b(N), c(N)
    integer :: N, i

!$omp target map(to: a,b) map(from: c)
!$omp teams distribute parallel do 
    do i=1,N
        c(i) = a(i) * b(i)
    end do
!$omp end target
end subroutine

#elif defined(VMUL1)

subroutine vmul(a, b, c, N)
    implicit none
    integer :: a(N), b(N), c(N)
    integer :: N, i

!$omp target map(to: a,b) map(from: c)
    do i=1,N
        c(i) = a(i) * b(i)
    end do
!$omp end target
end subroutine

#elif defined(VMUL2)

subroutine vmul(a, b, c, N)
    implicit none
    integer :: a(N), b(N), c(N)
    integer :: N

!$omp target map(to: a,b) map(from: c)
        c(1) = a(1) * b(1)
!$omp end target
end subroutine

#elif defined(VMUL3)

subroutine vmul(a, b, c, N)
    implicit none
    integer :: a(N), b(N), c(N)
    integer :: N, x, y, z
    x = a(1)
    y = b(1)

!$omp target map(from: z)
        z = x * y
!$omp end target
    c(1) = z
end subroutine

#else
#error no VMULx defined
#endif

program vmuldemo
    implicit none
    integer, parameter :: N = 100000
    integer a(N), b(N), c(N), validate(N)
    integer i, num, flag;
    num = N
    
    flag = -1
    do i = 1, N
        a(i) = i+1;
        b(i) = i+2;
        validate(i) = a(i) * b(i);
    enddo

    call vmul(a, b, c, N)

#if defined(VMUL2) || defined(VMUL3)
    num = 1
#endif
    do i = 1, num
        if (c(i) .ne. validate(i)) then
            ! print 1st bad index
            if ( flag .eq. -1 ) then
                write(*, '(A, I0, A, I0, A, I0, A, I0)') "First fail: c(", i, "):", c(i), " != validate(", i, "):", validate(i)
            endif
            flag = i;
        endif
    enddo
    if (flag .eq. -1) then
        write(6,*) "Success"
        call exit(0)
    else
        write(*, '(A, I0, A, I0, A, I0, A, I0)') "Last fail:  c(", flag, "):", c(flag), " != validate(", flag, "):", validate(flag)
        write(6,*) "FAILED"
        call exit(1)
    endif
end program


