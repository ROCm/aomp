program test
    implicit none
    real,allocatable,dimension(:,:) :: A,B
    real,allocatable,dimension(:)   :: C
    integer                         :: i,j,k
    integer,parameter               :: M = 1024*1024, N = 256, niter = 10

    allocate(A(N,M),B(N,M),C(M))
    !$omp target enter data map(alloc:A,B,C)
    do k=1,niter
        !$omp target teams distribute parallel do num_teams(960) num_threads(128)
        do i = 1,M
            C(i) = i
            do j=1,N
                B(j,i) = j + (C(i)-1) * M
                A(j,i) = B(j,i)
            enddo
        enddo
        !$omp end target teams distribute parallel do
    enddo
    !$omp target update from(A)
    write(*,*) "A(1,1) = ", A(1,1)
    if (a(1,1) .ne. 1.0) stop 2
    print *, "PASS"
    return
end program test
