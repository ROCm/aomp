
      module precision_mod
      implicit none
      integer, parameter :: dp = selected_real_kind(15,100)
      integer, parameter :: sp = selected_real_kind(6,30)
!     --------------------------------
!     use wp = sp for single precision
!     use wp = dp for double precision
!     --------------------------------
      integer, parameter :: wp = dp

      integer, parameter :: i8 = selected_int_kind( 12 )
      end module precision_mod

      module gemm_mod
      use precision_mod
      contains


      subroutine test_gemm_nn_strided_batched(m,n,k,batchCount,use_gpu0)
      use precision_mod
      implicit none
      integer, intent(in) :: m,n,k,batchCount
      logical, intent(in) :: use_gpu0

      real(kind=wp), allocatable :: A(:,:,:)
      real(kind=wp), allocatable :: B(:,:,:)
      real(kind=wp), allocatable :: C(:,:,:)
      real(kind=wp), allocatable :: Chost(:,:,:)
      real(kind=wp) :: alpha,beta
      integer :: lda,ldb,ldc

      integer, parameter :: idebug = 1
      real(kind=wp), parameter :: tol  = 1e-5

      real(kind=wp) :: max_err, cnorm_1,cnorm_2,cnorm_max
      real(kind=wp) :: c_ij, ch_ij
      integer :: i,i1,i2,i3
      integer(kind=i8) :: strideA,strideB,strideC
      integer :: ic,jc,ib,ibatch

      logical :: use_gpu

      use_gpu = use_gpu0

      allocate( A(m,k,batchCount) )
      allocate( B(k,n,batchCount) )
      allocate( C(m,n,batchCount) )
      allocate( Chost(m,n,batchCount) )

      lda = size(A,1)
      ldb = size(B,1)
      ldc = size(C,1)

      A = 1
      B = 2
      C = 3

      Chost = C
      alpha = -1
      beta = 1

!     -------------------
!     perform on cpu host
!     -------------------
!$omp parallel do
      do i=1,batchCount
        Chost(1:m,1:n,i) = beta * Chost(1:m,1:n,i) +                     &
     &                 alpha*matmul(A(1:m,1:k,i),B(1:k,1:n,i))
      enddo

      strideA = size(A,1)*size(A,2)
      strideB = size(B,1)*size(B,2)
      strideC = size(C,1)*size(C,2)

!$omp  target data                                                        &
!$omp& if (use_gpu)                                                       &
!$omp& map(to:m,n,k,batchCount,lda,ldb,ldc)                               &
!$omp& map(to:alpha,beta,strideA,strideB,strideC)                         &
!$omp& map(to:A,B) map(C)

!$omp  target teams distribute parallel do simd                              &
!$omp& if (use_gpu)                                                          &
!$omp& private(ic,jc,ib,c_ij)

      do ibatch=1,batchCount

        do jc=1,n
        do ic=1,m
          c_ij = 0
          do ib=1,k
           c_ij = c_ij + A(ic,ib,ibatch)*B(ib,jc,ibatch)
          enddo
          if (beta.eq.0) then
             C(ic,jc,ibatch) = alpha * c_ij
          else
             C(ic,jc,ibatch) = beta*C(ic,jc,ibatch) + alpha*c_ij
          endif
        enddo
        enddo
      enddo

!$omp end target data

       max_err = 0
       cnorm_max = 0
       cnorm_1 = 0
       cnorm_2 = 0
!$omp  parallel do                                                       &
!$omp& private(c_ij,ch_ij)                                               &
!$omp& reduction(max:max_err,cnorm_max)                                  &
!$omp& reduction(+:cnorm_1,cnorm_2)
       do i3=1,size(C,3)
       do i2=1,size(C,2)
       do i1=1,size(C,1)
         c_ij = C(i1,i2,i3)
         ch_ij = Chost(i1,i2,i3)

         max_err = max(max_err,abs(c_ij - ch_ij))
         cnorm_max = max( cnorm_max, abs(ch_ij))
         cnorm_1 = cnorm_1 + abs(ch_ij)
         cnorm_2 = cnorm_2 + abs(ch_ij) * abs(ch_ij)
       enddo
       enddo
       enddo
       cnorm_2 = sqrt( cnorm_2 )

       print 9010, max_err, cnorm_max,cnorm_1,cnorm_2
 9010  format(' max_err= ',1pe14.4,' cnorm_max = ',1pe14.4,              &
     &        ' cnorm_1 = ',1pe14.4, ' cnorm_2 = ', 1pe14.4)

       if ((idebug >= 1) .and. (max_err > tol)) then
          do i3=1,size(C,3)
          do i2=1,size(C,2)
          do i1=1,size(C,1)
           c_ij = C(i1,i2,i3)
           ch_ij = Chost(i1,i2,i3)
           if (abs(c_ij - ch_ij) > tol) then
            print 9100, i1,i2,i3,c_ij, i1,i2,i3,ch_ij
 9100       format(' C(',i4,',',i4,',',i4,') = ',1pe14.4,                &
     &             ' Ch(',i4,',',i4,',',i4,') = ',1pe14.4 )
            stop 2
           endif
          enddo
          enddo
          enddo
       endif

       deallocate(A,B,C,Chost)
       return
       end subroutine test_gemm_nn_strided_batched


       end module gemm_mod


      program main_gemm_nn_strided_batched
!$    use omp_lib
      use precision_mod
      use gemm_mod
      implicit none

      integer :: m,n,k,batchCount
      integer :: nthreads
      logical :: use_gpu

      nthreads = 1
!$omp parallel
!$omp master
!$    nthreads = omp_get_num_threads()
!$omp end master
!$omp end parallel
      print 9010,nthreads
 9010 format(' nthreads = ', i6)

      m = 3
      n = 3
      k = 3
      batchCount = 64

      use_gpu = .false.
      print*,'use_gpu ',use_gpu
      print 9020, m,n,k,batchCount
 9020 format(' m,n,k,batchCount ', 4(1x,i6) )
      call test_gemm_nn_strided_batched(m,n,k,batchCount,use_gpu)

      use_gpu = .true.
      print*,'use_gpu ',use_gpu
      print 9030, m,n,k,batchCount
 9030 format(' m,n,k,batchCount ', 4(1x,i6) )
      call test_gemm_nn_strided_batched(m,n,k,batchCount,use_gpu)

      m = 10
      n = 10
      k = 10
      batchCount = 2

      use_gpu = .false.
      print*,'use_gpu ',use_gpu
      print 9040, m,n,k,batchCount
 9040 format(' m,n,k,batchCount ', 4(1x,i6) )
      call test_gemm_nn_strided_batched(m,n,k,batchCount,use_gpu)

      use_gpu = .true.
      print*,'use_gpu ',use_gpu
      print 9050, m,n,k,batchCount
 9050 format(' m,n,k,batchCount ', 4(1x,i6) )
      call test_gemm_nn_strided_batched(m,n,k,batchCount,use_gpu)

      end program main_gemm_nn_strided_batched
