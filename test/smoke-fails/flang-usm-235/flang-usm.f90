program test
    use omp_lib
    implicit none
    integer,parameter                   :: nsize = 1024*256,ninc = 1024,niter = 10
    real,pointer                        :: a(:),b(:)
    integer                             :: i,j,k
    double precision                    :: wstart, wend, wstarttotal, wendtotal


    call omp_set_default_device(0)
 ! use -Mx,235,1    !$omp requires unified_shared_memory
    allocate(a(nsize*ninc),b(nsize*ninc))
    b = 1000
    !$omp target enter data map(alloc:a,b)
    wstarttotal = omp_get_wtime()
    do k=1,niter
    wstart = omp_get_wtime()
        !$omp target teams distribute parallel do collapse(2) !num_teams(880) thread_limit(256)
        do i=1,nsize
            do j=1,ninc
                a((i-1)*ninc+j) = j + 100*i + b((i-1)*ninc+j)
                !b((i-1)*ninc+j) = j + 1000*i + a((i-1)*ninc+j)
            end do
        enddo
    wend = omp_get_wtime()
    
    write(*,*) "Work took", 1e-9*(nsize*ninc*4)/(wEND - wSTART), "GB/s"
    enddo
    wendtotal = omp_get_wtime()
    !!$omp target update from(a,b)
    write(*,*) "a(2),b(2) =", a(2),b(2)
    write(*,*) "Work took", 1e-9*niter*(nsize*ninc*8)/(wENDtotal - wSTARTtotal), "GB/s"
    
end program test
