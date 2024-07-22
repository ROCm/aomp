program rename_func
    implicit none
    integer :: status, allstat
    allstat = 0

    status = rename('src', 'dst1')
    print *, 'status:', status
    allstat = allstat + status

    status = rename('dst1', 'dst2')
    print *, 'status:', status
    allstat = allstat + status

    status = rename('dst2', 'dst3')
    print *, 'status:', status
    allstat = allstat + status

    stop allstat
end program rename_func
