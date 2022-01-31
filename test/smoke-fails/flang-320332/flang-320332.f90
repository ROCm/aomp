program test
    integer :: i, iter, use_gpu
    real :: t_sum = 0, t_iter
#ifdef LEN_1
    character (len=1), dimension(:) :: array
#endif
#ifdef LEN_2
    character (len=2), dimension(:) :: array
#endif
#ifdef LEN_32
    character (len=32), dimension(:) :: array
#endif

#ifdef ENTER_DATA
    !$omp target enter data map(to:array)
#endif
#ifdef TARGET_DATA
    !$omp target data map(to:array)
#endif
    do i = 1,10
        print *, i
    end do
#ifdef TARGET_DATA
    !$omp end target data
#endif
end program test
