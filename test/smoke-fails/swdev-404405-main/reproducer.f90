program bandwidth

    use iso_c_binding
    use omp_lib
    use timer
    use allocator
    implicit none
    integer, parameter :: n = 110 * 1000000
    integer :: i, j, num_devices, nteams
    double precision :: GB
    double precision, pointer, dimension(:) :: a, b
    double precision :: t0, t1, elapsed
    integer(c_int) :: device_num = 0
    !$omp requires unified_shared_memory
    
    device_num =  omp_get_default_device()

    GB = 1000**3
    print *, "Data size (read and write):", (sizeof(t0) * n + sizeof(t0) * n) / GB, "GB"


    ! Fortran system allocator
    allocate(a(n))
    allocate(b(n))

    a = 1.0d0

    print *, "System allocator"
    call test(a, b, 10)

    deallocate(a)
    deallocate(b)

    print *, "hipMalloc"
    call hipmalloc(a, n)
    call hipmalloc(b, n)
    call test(a, b, 10)

    print *, "omp_target_alloc"
    call ompmalloc(a, n, device_num)
    call ompmalloc(b, n, device_num)
    call test(a, b, 10)

    contains

        subroutine test(a, b, niter)
            implicit none
            double precision, dimension(:) :: a, b
            integer :: niter
            a = 1.0d0
            b = 0.0d0
            do i=1,niter

                call tstart()
                !$omp target teams distribute parallel do private(j)
                do j=1,n
                    b(j) = a(j)
                end do
                call tstop()

                elapsed = telapsed()

                print *, "Elapsed:", elapsed, " s", " Bandwidth:", ( (sizeof(a) + sizeof(b)) / GB ) / elapsed, " GB/s"

            end do

            if (a(n) /= b(n)) then
                print *, "Error: a != b!", a(n), b(n)
            endif

        end subroutine

end program
