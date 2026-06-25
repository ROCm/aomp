program minimal_whole_array_ops
    implicit none
    integer, parameter :: N = 64, M = 2
    real(8) :: A(N,M), B(N,M), s(N), out(N,M)
    integer :: i, k, nerr

    do i = 1, N
        s(i) = 0.5d0 + 0.001d0*real(i,8)
        do k = 1, M
            A(i,k) = real(i+k,8)*0.01d0
            B(i,k) = real(i*k,8)*0.001d0
        end do
    end do

    !$omp target teams distribute parallel do map(to:A,B,s) map(from:out)
    do i = 1, N
        out(i,:) = A(i,:) + s(i)*(B(i,:) - A(i,:))
    end do
    !$omp end target teams distribute parallel do

    nerr = 0
    do i = 1, N
        do k = 1, M
            if (abs(out(i,k) - (A(i,k) + s(i)*(B(i,k) - A(i,k)))) > 1d-12) nerr = nerr + 1
        end do
    end do
    if (nerr == 0) then
        print *, "PASS"
    else
        print *, "FAIL:", nerr, "of", N*M
    end if
end program
