program minimal_array_constructor
    implicit none
    integer, parameter :: N = 64
    real(8) :: x(N), y(N), z(N), out(N,3)
    real(8) :: loc(3)
    integer :: i, nerr

    do i = 1, N
        x(i) = real(i,8)*0.1d0; y(i) = real(i,8)*0.2d0; z(i) = real(i,8)*0.3d0
    end do

    !$omp target teams distribute parallel do map(to:x,y,z) map(from:out) private(loc)
    do i = 1, N
        loc = [x(i), y(i), z(i)]
        out(i,1) = loc(1); out(i,2) = loc(2); out(i,3) = loc(3)
    end do
    !$omp end target teams distribute parallel do

    nerr = 0
    do i = 1, N
        if (abs(out(i,1)-x(i)) > 1d-14 .or. &
            abs(out(i,2)-y(i)) > 1d-14 .or. &
            abs(out(i,3)-z(i)) > 1d-14) nerr = nerr + 1
    end do
    if (nerr == 0) then
        print *, "PASS"
    else
        print *, "FAIL:", nerr, "of", N
    end if
end program
