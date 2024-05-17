program test_omp
    implicit none
    integer :: m,n,k,batchCount
    real, allocatable :: A(:,:,:), B(:,:,:),C(:,:,:)
    real :: csum

    integer :: i1,i2,i3

    m = 5
    n = 6
    k = 7
    batchCount = 16
    allocate( C(m,n,batchCount) )
    allocate( A(m,k,batchCount) )
    allocate( B(k,n,batchCount) )

    !$omp target data map(A,B,C)

    !$omp target teams distribute parallel do SIMD
    do i3=1,batchCount
    do i2=1,size(C,2)
    C(1:m,i2,i3) = 1
    enddo
    enddo


    !$omp target teams distribute   parallel do SIMD
    do i3=1,size(A,3)
    do i2=1,size(A,2)
    do i1=1,size(A,1)
    A(i1,i2,i3) = i1 + (i2-1)*size(A,1) +                          &
        &       (i3-1)*size(A,1)*size(A,2)
    enddo
    enddo
    enddo

    !$omp target teams distribute   parallel do SIMD
    do i3=1,size(B,3)
    do i2=1,size(B,2)
    do i1=1,size(B,1)
    B(i1,i2,i3) = 1.0/( i1+(i2-1)*size(B,1) +                      &
        &                        (i3-1)*size(B,1)*size(B,2))
    enddo
    enddo
    enddo

    !$omp target teams distribute parallel do SIMD
    do i3=1,batchCount
    C(1:m,1:n,i3) = C(1:m,1:n,i3) +                                  &
        &         matmul( A(1:m,1:k,i3), B(1:k,1:n,i3))
    enddo

    !$omp end target data

    csum = 0
    !$omp parallel do collapse(3) reduction(+:csum)
    do i3=1,size(C,3)
    do i2=1,size(C,2)
    do i1=1,size(C,1)
    csum = csum + abs(C(i1,i2,i3))
    enddo
    enddo
    enddo

    print*,'m,n,k,batchCount',m,n,k,batchCount
    print*,'csum = ', csum

    stop
end program test_omp

