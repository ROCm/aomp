program omp_subroutine
    implicit none
    integer, parameter :: N = 1
    double precision, allocatable, target, dimension(:) :: c


    allocate(c(N))
    ! Expected output c(1) = 1
    ! Wrong output on flang and cce
    call test1(c)
    write (*,*) "test1:", c

    ! Wrong output on flang, correct output on cce
    !$omp target enter data map(to:c)
    call test2(c)
    !$omp target update from(c)
    write (*,*) "test2:", c

    ! Correct output on flang
    call test3()

    ! Segmentation fault on flang
    call test4(c)
    write (*,*) "test4:", c


contains
    subroutine test1(c)
        double precision :: c(:)
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(1) = 1.0
        !$omp end target data
        !$omp target update from(c)
    end subroutine


    subroutine test2(c)
        double precision :: c(:)
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
        double precision, allocatable, target, dimension(:) :: c
        deallocate(c)
        allocate(c(N))
        !$omp target enter data map(to:c)
        !$omp target data use_device_ptr(c)
        c(1) = 1.0
        !$omp end target data
        !$omp target update from(c)
    end subroutine

end program omp_subroutine
