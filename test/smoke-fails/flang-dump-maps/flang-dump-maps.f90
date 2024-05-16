program test
    implicit none
  INTERFACE
    SUBROUTINE ompx_dump_mapping_tables() BIND(C)
    END SUBROUTINE ompx_dump_mapping_tables
  END INTERFACE
    integer :: i,j,niters,N
    real,dimension(:),allocatable :: A,B
    !$omp requires unified_shared_memory

    N = 1024*1024
    niters = 100
    allocate(A(N),B(N))
    !$omp target enter data map(alloc:A,B)
    call ompx_dump_mapping_tables()
        !$omp target teams distribute parallel do num_threads(64)
        do i=1,N
            A(i) = i-2+j
            B(i) = A(i) * 2
        enddo
    !$omp target exit data map(delete:A,B)
end program test
