program atest
    implicit none
    integer, parameter :: N = 1000
    integer :: a(N)
    integer :: i, val, errors
    val = 2

!$omp target teams distribute parallel do num_teams(val) map(from:a)
    do i=1,N
        a(i) = i
    end do

    errors = 0
    do i = 1, N
        if (a(i) .ne. i) then
            errors = errors + 1
        endif
    enddo
    if (errors .eq. 0) then
        print *, "Success"
        call exit(0)
    else
        print *, "FAILED", errors
        call exit(1)
    endif
end program
