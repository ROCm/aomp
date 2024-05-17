program test
    implicit none
    integer :: i,j,niters,N
    real,dimension(:),allocatable :: A,B
    !$omp requires unified_shared_memory

    N = 1024*1024
    niters = 100
    allocate(A(N),B(N))
    !$omp target enter data map(alloc:A,B)
    do j=1,niters
        !$omp target teams distribute parallel do num_threads(64)
        do i=1,N
            A(i) = i-2+j
            B(i) = A(i) * 2
        enddo
    enddo
    !$omp target exit data map(delete:A,B)
    write(*,*) "A(3),B(3) = ",A(3),B(3)
    if (a(3) .ne. 101 .or.  b(3) .ne. 202) then
       write(*,*) "Failed"
       stop 2
    endif
end program test
