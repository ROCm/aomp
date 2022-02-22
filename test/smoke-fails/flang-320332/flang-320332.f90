program test
    integer :: i, arr_size = 10
    character (len=32), dimension(:), allocatable :: array

    allocate(array(arr_size))
    array(:) = "test"

    !$omp target enter data map(to:array)

    do i = 1,arr_size
        print *, array(i)
    end do
end program test
