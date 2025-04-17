program foo
    implicit none
    integer :: i
    i = 42
    !$omp target if(.false.)
        i = 21
    !$omp end target
    if (i.ne.42) then
        print '("Got ",I," but expected ",I)', i, 42
        stop 1
    end if
end program foo
