program omp_subroutine
    implicit none
    integer, parameter :: N = 1
    double precision, allocatable, target, dimension(:) :: c


    allocate(c(N))
    ! Expected output c(1) = 1
    ! Wrong output on flang and cce
    call test1(c)
    if (c(1) .ne. 1.0) then
       write (*,*) "FAIL : test1:", c
       stop 2
    endif


    ! Wrong output on flang, correct output on cce
    !$omp target enter data map(to:c)
    call test2(c)
    !$omp target update from(c)
    if (c(1) .ne. 1.0) then
       write (*,*) "FAIL : test2:", c
       stop 2
    endif

    ! Correct output on flang
    call test3()
    if (c(1) .ne. 1.0) then
       write (*,*) "FAIL : test3:", c
       stop 2
    endif

    ! Segmentation fault on flang
    call test4(c)
    if (c(1) .ne. 1.0) then
       write (*,*) "FAIL : test4:", c
       stop 2
    endif

    print *, "PASS"
    return

contains
    subroutine test1(c)
        double precision, intent(inout) :: c(:)
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(1) = 1.0
        !$omp end target data
        !$omp target update from(c)
    end subroutine


    subroutine test2(c)
        double precision, intent(inout) :: c(:)
        !$omp target data use_device_ptr(c)
        c(1) = 1.0
        !$omp end target data
    end subroutine


    subroutine test3()
        double precision, allocatable, target, dimension(:) :: c
        allocate(c(N))
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(1) = 1.0
        !$omp end target data
        !$omp target update from(c)
        write (*,*) "test3:", c
    end subroutine


    subroutine test4(c)
        double precision, allocatable, target, dimension(:), intent(inout) :: c
        deallocate(c)
        allocate(c(N))
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(1) = 1.0
        !$omp end target data
        !$omp target update from(c)
    end subroutine

end program omp_subroutine
