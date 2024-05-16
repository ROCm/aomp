module gbtr_mod
    integer, parameter :: dp = selected_real_kind(15,100)
    integer, parameter :: wp = dp
    integer, parameter :: i8 = selected_int_kind(6)
contains
    subroutine getrf_npvt_3x3(m,n,A,lda)
        implicit none
        !$omp declare target 
        integer, intent(in) :: m,n,lda
        real(kind=wp), intent(inout) :: A(lda,*)
        integer, parameter :: idebug = 0
        real(kind=wp) :: U11, inv_U11, U22
        integer :: i,j
        real(kind=wp) :: err, max_err
        U11 = A(1,1);
        inv_U11 = 1/U11;
        A(2,1) = A(2,1) * inv_U11;
        A(3,1) = A(3,1) * inv_U11;

        A(2,2) = A(2,2) - A(2,1)*A(1,2);
        A(3,2) = A(3,2) - A(3,1)*A(1,2);
        A(2,3) = A(2,3) - A(2,1)*A(1,3);
        A(3,3) = A(3,3) - A(3,1)*A(1,3);

        U22 = A(2,2);
        A(3,2) = A(3,2)/U22;

        A(3,3) = A(3,3) - A(3,2)*A(2,3);
        return
    end subroutine getrf_npvt_3x3
    subroutine getrf_npvt( m, n, A, lda)
        implicit none
        !$omp declare target 
        integer, intent(in) :: m,n,lda
        real(kind=wp), intent(inout) :: A(lda,*)
        integer :: min_mn
        integer :: j,jp1,ia
        real(kind=wp) :: Ujj, inv_Ujj
        logical :: is_special_case
        min_mn = min(m,n)
        is_special_case = (m.eq.3).and.(n.eq.3)
        if (is_special_case) then
            call getrf_npvt_3x3(m,n,A,lda)
            return
        endif
        do j=1,min_mn
        jp1 = j + 1
        Ujj = A(j,j)
        inv_Ujj = 1/Ujj
        do ia=jp1,m
        A(ia,j) = A(ia,j) * inv_Ujj
        enddo
        A(jp1:m,jp1:n) = A(jp1:m,jp1:n) -                               &
            &                        matmul(A(jp1:m,j:j),A(j:j,jp1:n))
        enddo
        return
    end subroutine getrf_npvt
    subroutine getrs_npvt_3x3( n,nrhs, LU, ld, B, ldb )
        implicit none
        !$omp declare target 
        integer, intent(in) :: n,nrhs,ld,ldb
        real(kind=wp), intent(in) :: LU(ld,*)
        real(kind=wp), intent(inout) :: B(ldb,nrhs)
        integer :: k
        real(kind=wp) :: inv_LU11,inv_LU22,inv_LU33
        do k=1,nrhs
        B(2,k) = B(2,k) - LU(2,1)*B(1,k)
        B(3,k) = B(3,k) - LU(3,1)*B(1,k) - LU(3,2)*B(2,k)
        enddo

        inv_LU11 = 1/LU(1,1)
        inv_LU22 = 1/LU(2,2)
        inv_LU33 = 1/LU(3,3)
        do k=1,nrhs
        B(3,k) = B(3,k) * inv_LU33 
        B(2,k) = (B(2,k) - LU(2,3)*B(3,k)) * inv_LU22 
        B(1,k) = (B(1,k) - LU(1,2)*B(2,k) - LU(1,3)*B(3,k)) * inv_LU11 
        enddo
        return
    end subroutine getrs_npvt_3x3
    subroutine getrs_npvt( n,nrhs, A,lda, B,ldb )
        implicit none
        !$omp declare target 
        integer, intent(in) :: n,nrhs,lda,ldb
        real(kind=wp), intent(in) :: A(lda,*)
        real(kind=wp), intent(inout) :: B(ldb,nrhs)
        logical :: is_special_case
        integer :: i,j,k, ir
        real(kind=wp) :: Uii, inv_Uii
        is_special_case = (n.eq.3)
        if (is_special_case) then
            call getrs_npvt_3x3(n,nrhs,A,lda,B,ldb)
            return
        endif
        do i=1,n
        do j=1,(i-1)
        do k=1,nrhs
        B(i,k) = B(i,k) - A(i,j) * B(j,k)
        enddo
        enddo
        enddo
        do ir=1,n
        i = n - ir + 1
        do j=(i+1),n
        do k=1,nrhs
        B(i,k) = B(i,k) - A(i,j)*B(j,k)
        enddo
        enddo
        Uii = A(i,i)
        inv_Uii = 1/Uii
        do k=1,nrhs
        B(i,k) = B(i,k)  * inv_Uii
        enddo
        enddo
        return
    end subroutine getrs_npvt
    subroutine gbtrf_npvt(nb,nblocks, A,lda,B,ldb,C,ldc)
        implicit none
        !$omp declare target
        integer, intent(in) :: nb,nblocks,lda,ldb,ldc
        real(kind=wp), intent(inout) :: A(lda,nb,nblocks)
        real(kind=wp), intent(inout) :: B(ldb,nb,nblocks)
        real(kind=wp), intent(inout) :: C(ldc,nb,nblocks)
        integer :: ldu, ldd, k 
        integer :: mm,nn,nrhs
        ldu = ldc
        ldd = ldb
        k = 1
        mm = nb
        nn = nb
        call getrf_npvt( mm, nn, B(1,1,k), ldd )

        do k=1,(nblocks-1)
        nn = nb
        nrhs = nb
        call getrs_npvt( nn,nrhs, B(1,1,k), ldd, C(1,1,k),ldc )

        B(1:nb,1:nb,k+1) = B(1:nb,1:nb,k+1) -                           &
            &        matmul( A(1:nb,1:nb,k+1), C(1:nb,1:nb,k) )
        mm = nb
        nn = nb
        call getrf_npvt( mm,nn,B(1,1,k+1),ldd )
        enddo

        return
    end subroutine gbtrf_npvt
    subroutine gbtrf_npvt_strided_batched( nb,nblocks, batchCount,     &
            &               A,lda, strideA, B, ldb, strideB, C, ldc, strideC)
        implicit none
        integer, intent(in) :: nb, nblocks, batchCount
        integer, intent(in) :: lda,ldb,ldc
        integer, intent(in) :: strideA, strideB, strideC
        real(kind=wp), intent(inout) :: A(*)
        real(kind=wp), intent(inout) :: B(*)
        real(kind=wp), intent(inout) :: C(*)
        integer(kind=i8) :: idxA,idxB,idxC
        integer(kind=i8) :: lstrideA,lstrideB,lstrideC
        integer :: i
        integer(kind=i8) :: sizeA,sizeB,sizeC
        lstrideA = strideA
        lstrideB = strideB
        lstrideC = strideC
        sizeA = lstrideA * batchCount
        sizeB = lstrideB * batchCount
        sizeC = lstrideC * batchCount
        #ifdef SHOW_BUG
        !$omp  target data                                                       &
        !$omp& map(to:nb,nblocks,batchCount,lda,ldb,ldc)                         &
        !$omp& map(to:lstrideA,lstrideB,lstrideC)                                &
        !$omp& map(A(1:sizeA),B(1:sizeB),C(1:sizeC))
        #else
        !$omp  target data                                                       &
        !$omp& map(to:nb,nblocks,batchCount,lda,ldb,ldc)                         &
        !$omp& map(to:lstrideA,lstrideB,lstrideC)                                 
        !!$omp& map(tofrom:A(1:sizeA),B(1:sizeB),C(1:sizeC))
        #endif

        !$omp target teams distribute parallel do SIMD                           &
        !$omp& private(idxA,idxB,idxC)                                           &
        !$omp& map(A(1:sizeA),B(1:sizeB),C(1:sizeC))
        do i=1,batchCount
        idxA = lstrideA * (i-1) + 1
        idxB = lstrideB * (i-1) + 1
        idxC = lstrideC * (i-1) + 1
        call gbtrf_npvt(nb,nblocks,A(idxA),lda,B(idxB),ldb,C(idxC),ldc)
        enddo
        !$omp end target data
        return
    end subroutine gbtrf_npvt_strided_batched
end module gbtr_mod
