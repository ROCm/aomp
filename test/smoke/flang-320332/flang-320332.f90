program test
    integer :: i, arr_size = 10
    character (len=32), dimension(:), allocatable :: array

    allocate(array(arr_size))
    array(:) = "test"

    !$omp target enter data map(to:array)

    do i = 1,arr_size
        if ( array(i) .ne. "test" ) then
          print *, "FAILED! Expected answer : test instead of ", array(i)
          stop 2
        endif
    end do
    print *, "SUCCESS"
    return
end program test
