!---------------------------------------------------------------------
!---------------------------------------------------------------------
!
!  bt_data module
!
!---------------------------------------------------------------------
!---------------------------------------------------------------------
 
      module bt_data

!---------------------------------------------------------------------
! The following include file is generated automatically by the
! "setparams" utility. It defines 
!      maxcells:      the square root of the maximum number of processors
!      problem_size:  12, 64, 102, 162 (for class T, A, B, C)
!      dt_default:    default time step for this problem size if no
!                     config file
!      niter_default: default number of iterations for this problem size
!---------------------------------------------------------------------

      include 'npbparams.h'

      integer           aa, bb, cc, BLOCK_SIZE
      parameter        (aa=1, bb=2, cc=3, BLOCK_SIZE=5)

      integer           grid_points(3)
      double precision  elapsed_time
!!$acc declare create(grid_points)
!!$omp declare target to(grid_points)

      double precision  tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3,  &
     &                  dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4,  &
     &                  dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt,  &
     &                  ce(5,13), dxmax, dymax, dzmax, xxcon1, xxcon2,  &
     &                  xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,  &
     &                  dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,  &
     &                  yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,  &
     &                  zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1,  &
     &                  dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1,  &
     &                  dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2,  &
     &                  c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1,  &
     &                  dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1,  &
     &                  c2dtty1, c2dttz1, comz1, comz4, comz5, comz6,  &
     &                  c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16

      integer IMAX, JMAX, KMAX
      parameter (IMAX=problem_size,JMAX=problem_size,KMAX=problem_size)

!
!   field arrays 
!
      double precision, allocatable ::  &
     &   us      (:, :, :),  &
     &   vs      (:, :, :),  &
     &   ws      (:, :, :),  &
     &   qs      (:, :, :),  &
     &   rho_i   (:, :, :),  &
     &   square  (:, :, :),  &
     &   forcing (:, :, :, :),  &
     &   u       (:, :, :, :),  &
     &   rhs     (:, :, :, :)
!!$acc declare create(u, us, vs, ws, qs, rho_i,  &
!!$acc&               square, rhs, forcing)
!!$omp declare target to(u, us, vs, ws, qs, rho_i,  &
!!$omp&               square, rhs, forcing)
      double precision cuf(0:problem_size),   q  (0:problem_size),  &
     &                 ue (0:problem_size,5), buf(0:problem_size,5)

!
!-----------------------------------------------------------------------
!   Timer constants
!-----------------------------------------------------------------------
      integer t_rhsx, t_rhsy, t_rhsz, t_xsolve, t_ysolve, t_zsolve,  &
     &        t_rdis1, t_solsub, t_add, t_rhs, t_last, t_total
      parameter (t_total = 1)
      parameter (t_rhsx = 2)
      parameter (t_rhsy = 3)
      parameter (t_rhsz = 4)
      parameter (t_rhs = 5)
      parameter (t_xsolve = 6)
      parameter (t_ysolve = 7)
      parameter (t_zsolve = 8)
      parameter (t_rdis1 = 9)
      parameter (t_solsub = 10)
      parameter (t_add = 11)
      parameter (t_last = 11)

      logical timeron

      end module bt_data


!---------------------------------------------------------------------
!---------------------------------------------------------------------
!
!  work_lhs module
!
!---------------------------------------------------------------------
!---------------------------------------------------------------------

      module work_lhs

      use bt_data, only : problem_size

      integer :: bsize
!!$acc declare create(bsize)
!!$omp declare target to(bsize)

      integer, parameter :: ndim = problem_size - 1
      integer, parameter :: ndim1 = ndim - 1

      double precision, allocatable ::  &
     &   fjac(:, :, :, :, :),  &
!     >   njac(:, :, :, :, :),
     &   lhsa(:, :, :, :, :),  &
     &   lhsb(:, :, :, :, :),  &
     &   lhsc(:, :, :, :, :),  &
     &   rhsx(:, :,    :, :)
!!$acc declare create(fjac, lhsa, lhsb, lhsc, rhsx)
!!$omp declare target to(fjac, lhsa, lhsb, lhsc, rhsx)
      external matvec_sub, matmul_sub, binvcrhs, binvrhs, lhsinit
!!$acc routine(matvec_sub) vector
!!$acc routine(matmul_sub) vector
!!$acc routine(binvcrhs) vector
!!$acc routine(binvrhs) vector
!!$acc routine(lhsinit) vector
!!$omp declare target (matvec_sub)
!!$omp declare target (matmul_sub) 
!!$omp declare target (binvcrhs) 
!!$omp declare target (binvrhs) 
!!$omp declare target (lhsinit) 


      end module work_lhs


!---------------------------------------------------------------------
!---------------------------------------------------------------------

      subroutine lhsinit(ni, k)

!---------------------------------------------------------------------
!---------------------------------------------------------------------
      use work_lhs, only : lhsa, lhsb, lhsc, bsize
      implicit none
!!$acc routine vector
!!$omp declare target

      integer ni, k

      integer i, m, n, j

!---------------------------------------------------------------------
!     zero the whole left hand side for starters
!     set all diagonal values to 1. This is overkill, but convenient
!---------------------------------------------------------------------
!!$acc loop seq
      do i = 0, ni, ni
!!$acc loop vector
!!!$omp target teams distribute parallel do collapse(3)
         do j = 1, bsize
!!$acc loop seq
            do m = 1, 5
!!$acc loop seq
               do n = 1, 5
                  lhsa(j,m,n,i,k) = 0.0d0
                  lhsb(j,m,n,i,k) = 0.0d0
                  lhsc(j,m,n,i,k) = 0.0d0
               end do
               lhsb(j,m,m,i,k) = 1.0d0
            end do
         end do
      end do

      return
      end subroutine
      

!---------------------------------------------------------------------
!---------------------------------------------------------------------

      subroutine alloc_space

!---------------------------------------------------------------------
!---------------------------------------------------------------------

!---------------------------------------------------------------------
! allocate space dynamically for data arrays
!---------------------------------------------------------------------

      use bt_data
      implicit none

      integer ios

      integer IMAXP, JMAXP
      parameter (IMAXP=IMAX/2*2,JMAXP=JMAX/2*2)

!
!   to improve cache performance, grid dimensions padded by 1 
!   for even number sizes only.
!
      allocate (  &
     &   us      (0:IMAXP, 0:JMAXP, 0:KMAX-1),  &
     &   vs      (0:IMAXP, 0:JMAXP, 0:KMAX-1),  &
     &   ws      (0:IMAXP, 0:JMAXP, 0:KMAX-1),  &
     &   qs      (0:IMAXP, 0:JMAXP, 0:KMAX-1),  &
     &   rho_i   (0:IMAXP, 0:JMAXP, 0:KMAX-1),  &
     &   square  (0:IMAXP, 0:JMAXP, 0:KMAX-1),  &
     &   forcing (0:IMAXP, 0:JMAXP, 0:KMAX-1,5),  &
     &   u       (0:IMAXP, 0:JMAXP, 0:KMAX-1,5),  &
     &   rhs     (0:IMAXP, 0:JMAXP, 0:KMAX-1,5),  &
     &         stat = ios)

      if (ios .ne. 0) then
         write(*,*) 'Error encountered in allocating space'
         stop
      endif

      return
      end

!---------------------------------------------------------------------
!---------------------------------------------------------------------

      subroutine alloc_lhs_space()

!---------------------------------------------------------------------
!---------------------------------------------------------------------
      use work_lhs
      implicit none

      integer ios

      allocate (  &
     &   fjac(ndim1, 5, 5, 0:ndim, ndim-1),  &
!     >   njac(ndim1, 5, 5, 0:ndim, ndim-1),
     &   lhsa(ndim1, 5, 5, 0:ndim, ndim-1),  &
     &   lhsb(ndim1, 5, 5, 0:ndim, ndim-1),  &
     &   lhsc(ndim1, 5, 5, 0:ndim, ndim-1),  &
     &   rhsx(ndim1, 5,    0:ndim, ndim-1),  &
     &         stat = ios)

      if (ios .ne. 0) then
         write(*,*) 'Error encountered in allocating space'
         stop
      endif
      write(*,*) "Passed"
      return
      end


      program repro
         call alloc_lhs_space()
      end program repro
