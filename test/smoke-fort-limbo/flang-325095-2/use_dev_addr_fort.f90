program omp_subroutine
    implicit none
    integer, parameter :: N = 3
    double precision, allocatable, dimension(:) :: c

    allocate(c(N))
    c = -42.0

    call test0(c)
    if (c(3) .ne. 1.0) then
       write (*,*) "FAIL : test0:", c
       stop 2
    endif

    print *, c
    c(3) = -42.0

    ! Expected output c(3) = 1
    call test1(c)
    if (c(3) .ne. 1.0) then
       write (*,*) "FAIL : test1:", c
       stop 2
    endif

    print *, c
    c(3) = -42.0

    !$omp target enter data map(to:c)
    call test2(c)
    !$omp target update from(c)
    if (c(3) .ne. 1.0) then
       write (*,*) "FAIL : test2:", c
       stop 2
    endif

    print *, c
    c(3) = -42.0

    call test3()
    if (c(3) .ne. -42.0) then
       write (*,*) "FAIL : test3:", c
       stop 2
    endif

    print *, c
    c(3) = -42.0

    call test4(c)
    if (c(3) .ne. 1.0) then
       write (*,*) "FAIL : test4:", c
       stop 2
    endif

    print *, c
    print *, "PASS"
    return
contains
    subroutine test0(c)
        double precision, intent(inout) :: c(:)
        !$omp target data use_device_ptr(c) map(tofrom: c)
            c(3) = 1.0
        !$omp end target data
    end subroutine

    subroutine test1(c)
        double precision, intent(inout) :: c(:)
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(3) = 1.0
        !$omp end target data
        !$omp target update from(c)
    end subroutine

    subroutine test2(c)
        double precision, intent(inout) :: c(:)
        !$omp target data use_device_ptr(c)
        c(3) = 1.0
        !$omp end target data
    end subroutine

    subroutine test3()
        double precision, allocatable, dimension(:) :: c
        allocate(c(N))
       !$omp target enter data map(to:c)
       !$omp target data use_device_ptr(c)
            c(3) = 1.0
       !$omp end target data
       !$omp target update from(c)
        write (*,*) "test3:", c
    end subroutine


    subroutine test4(c)
        double precision, allocatable, dimension(:), intent(inout) :: c
        deallocate(c)
        allocate(c(N))
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(3) = 1.0
        !$omp end target data
        !$omp target update from(c)
    end subroutine

end program omp_subroutine