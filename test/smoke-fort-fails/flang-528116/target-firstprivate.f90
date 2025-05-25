program foo
    implicit none
    integer :: i
    i = 42
    ! if(.false.) is intentional to execute the target region on the host
    ! to see if it still adheres to OpenMP semantics and host execution
    !$omp target if(.false.)
        i = 21
    !$omp end target
    if (i.ne.42) then
        print '("Got ",I," but expected ",I)', i, 42
        stop 1
    end if
end program foo
