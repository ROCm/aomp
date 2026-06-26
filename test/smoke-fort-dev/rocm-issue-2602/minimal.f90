program t
    integer, parameter :: N = 64
    integer :: out(N, 2), i, nerr
    !$omp target teams distribute parallel do map(from:out)
    do i = 1, N
        out(i,:) = [i, i*2]
    end do
    !$omp end target teams distribute parallel do
    nerr = 0
    do i = 1, N
        if (out(i,1) /= i .or. out(i,2) /= i*2) nerr = nerr + 1
    end do
    if (nerr == 0) then
        print *, "PASS"
    else
        print *, "FAIL:", nerr, "of", N   ! without fix: FAIL: 31 of 64
    end if
end program
