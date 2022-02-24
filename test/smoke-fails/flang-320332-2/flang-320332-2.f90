program test
    integer :: i, iter, use_gpu
    integer, parameter :: arr_size = 10
    real :: t_sum = 0, t_iter
#ifdef LEN_1
    character (len=1), dimension(:), allocatable :: array
#endif
#ifdef LEN_2
    character (len=2), dimension(:), allocatable :: array
#endif
#ifdef LEN_32
    character (len=32), dimension(:), allocatable :: array
#endif

    allocate(array(arr_size))
    array(:) = "test"
#ifdef ENTER_DATA
    !$omp target enter data map(to:array)
#endif
#ifdef TARGET_DATA
    !$omp target data map(to:array)
#endif
    !$omp target data use_device_ptr(array)
    do i = 1,arr_size
        print *, array(i)
    end do
    !$omp end target data
#ifdef TARGET_DATA
    !$omp end target data
#endif
end program test
