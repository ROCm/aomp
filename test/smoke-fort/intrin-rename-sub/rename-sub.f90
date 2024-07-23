program rename_sub
    implicit none
    integer :: status, allstat
    allstat = 0

    status = -1
    call rename('src', 'dst1', status)
    print *, 'status:', status
    allstat = allstat + status

    ! without status
    call rename('dst1', 'dst2')
    print *, 'no status'

    status = -1
    call rename('dst2', 'dst3', status)
    print *, 'status:', status
    allstat = allstat + status

    stop allstat
end program rename_sub
