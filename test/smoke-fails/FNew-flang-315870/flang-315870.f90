! Copyright (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!

module m_random
   integer, parameter :: RKIND  = selected_real_kind(12)
   contains

   subroutine set_default_seeds()
        integer, dimension(34) :: seeds
        seeds(1)=290160
        seeds(2)=2556217
        seeds(3)=7161241
        seeds(4)=619582
        seeds(5)=2329578
        seeds(6)=431781
        seeds(7)=3632682
        seeds(8)=6355368
        seeds(9)=3900995
        seeds(10)=6516640
        seeds(11)=3315613
        seeds(12)=6114515
        seeds(13)=5730631
        seeds(14)=3339426
        seeds(15)=5857968
        seeds(16)=770915
        seeds(17)=4542500
        seeds(18)=4366800
        seeds(19)=7610154
        seeds(20)=5459821
        seeds(21)=7062514
        seeds(22)=177168
        seeds(23)=6052756
        seeds(24)=157396
        seeds(25)=2591103
        seeds(26)=5059992
        seeds(27)=1275832
        seeds(28)=980058
        seeds(29)=486656
        seeds(30)=131642
        seeds(31)=7330869
        seeds(32)=356226
        seeds(33)=2040732
        seeds(34)=806
        call random_seed(put=seeds)
   end subroutine set_default_seeds

   subroutine random_init()
    integer, dimension(:), pointer :: seeds
     INTEGER, DIMENSION(1:8) :: dt_seed
     integer :: n

     CALL RANDOM_SEED(size = n)
     ALLOCATE(seeds(n))
     CALL RANDOM_SEED()
     call set_default_seeds()
     CALL RANDOM_SEED(get = seeds)
     !print *, "seeds:"
     !write(*, '(I)') seeds
   end subroutine random_init

   subroutine random_1D(x)
     real (kind=RKIND) :: x(:)

     call random_number(x)
     x(:) = x(:)
     !write(*,'(F21.18)') x
   end subroutine random_1D

   subroutine random_1D_int(x, max_val)
     integer :: x(:)
     real (kind=RKIND), dimension(:), pointer :: rx  ! needed for Grell-Freitas convection scheme
     integer :: s(1)
     real (kind=RKIND):: max_val

     s = shape(x)
                    
     allocate(rx(s(1)))

     !call random_seed()
     call random_number(rx)
     x(:) = rx(:) * max_val
     !write(*,'(I5)') x
   end subroutine random_1D_int

   subroutine random_2D(x)
     real (kind=RKIND):: x(:,:)

     call random_number(x)
     x(:,:) = x(:,:)
     !write(*,'(F21.18)') x
   end subroutine random_2D

   subroutine random_2D_int(x, max_val)
     integer :: x(:,:)
     real (kind=RKIND), dimension(:,:), pointer :: rx  ! needed for Grell-Freitas convection scheme
     integer :: s(2)
     real (kind=RKIND) :: max_val
     INTEGER :: i, n
     INTEGER, DIMENSION(:), ALLOCATABLE :: seed

     s = shape(x)
                    
     allocate(rx(s(1), s(2)))

     call random_number(rx)
     x(:,:) = rx(:,:) * max_val
     !write(*,'(I5)') x
   end subroutine random_2D_int


   subroutine random_ele(y)
     real (kind=RKIND) :: y

     call random_number(y)
     !write(*,'(F21.18)') y
   end subroutine random_ele
end module m_random


module mpas_ut_func
   integer, parameter :: RKIND  = selected_real_kind(12)
   real (kind=RKIND), parameter :: pii     = 3.141592653589793   !< Constant: Pi
   real (kind=RKIND), parameter :: a       = 6371229.0           !< Constant: Spherical Earth radius [m]
   real (kind=RKIND), parameter :: omega   = 7.29212e-5          !< Constant: Angular rotation rate of the Earth [s-1]
   real (kind=RKIND), parameter :: gravity = 9.80616             !< Constant: Acceleration due to gravity [m s-2]
   real (kind=RKIND), parameter :: rgas    = 287.0               !< Constant: Gas constant for dry air [J kg-1 K-1]
   real (kind=RKIND), parameter :: rv      = 461.6               !< Constant: Gas constant for water vapor [J kg-1 K-1]
   real (kind=RKIND), parameter :: rvord   = rv/rgas             !
   real (kind=RKIND), parameter :: cp      = 7.*rgas/2.          !< Constant: Specific heat of dry air at constant pressure [J kg-1 K-1]
   real (kind=RKIND), parameter :: cv      = cp - rgas           !< Constant: Specific heat of dry air at constant volume [J kg-1 K-1]
   real (kind=RKIND), parameter :: cvpm    = -cv/cp              !
   real (kind=RKIND), parameter :: prandtl = 1.0                 !< Constant: Prandtl number
   real (kind=RKIND), parameter :: NMAX    = 10000000.0

   contains

   subroutine kernel1(nVertLevels, maxEdges, nCells, nEdges, nCellsSolve, cellStart, cellEnd, vertexStart, vertexEnd, edgeStart, edgeEnd, &
                                   cellSolveStart, cellSolveEnd, vertexSolveStart, vertexSolveEnd, edgeSolveStart, edgeSolveEnd, &
                                   dts, small_step, epssm, cf1, cf2, cf3, T_e2e, use_gpu&
                                   )
      use m_random
      implicit none


      !
      ! Dummy arguments
      !
      integer, intent(in) :: nVertLevels, maxEdges
      integer, intent(in) :: nCells, nEdges, nCellsSolve
      integer, intent(in) :: cellStart, cellEnd, vertexStart, vertexEnd, edgeStart, edgeEnd
      integer, intent(in) :: cellSolveStart, cellSolveEnd, vertexSolveStart, vertexSolveEnd, edgeSolveStart, edgeSolveEnd

      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: rho_zz
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: theta_m
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: ru_p
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: rw_p
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: rtheta_pp

      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: rtheta_pp_old
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: zz
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: exner
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: cqu
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: ruAvg
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: wwAvg
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: rho_pp
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: cofwt
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: coftz
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: zxu

      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: a_tri
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: alpha_tri
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: gamma_tri
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: dss
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: tend_ru
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: tend_rho
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: tend_rt
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: tend_rw
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: zgrid
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: cofwr
      real (kind=RKIND), dimension(nVertLevels,nCells+1) :: cofwz

      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: w
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: ru
      real (kind=RKIND), dimension(nVertLevels,nEdges+1) :: ru_save
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: rw
      real (kind=RKIND), dimension(nVertLevels+1,nCells+1) :: rw_save

      real (kind=RKIND), dimension(nVertLevels) :: fzm
      real (kind=RKIND), dimension(nVertLevels) :: fzp
      real (kind=RKIND), dimension(nVertLevels) :: rdzw
      real (kind=RKIND), dimension(nEdges+1) :: dcEdge
      real (kind=RKIND), dimension(nEdges+1) :: invDcEdge
      real (kind=RKIND), dimension(nCells+1) :: invAreaCell
      real (kind=RKIND), dimension(nVertLevels) :: cofrz
      real (kind=RKIND), dimension(nEdges+1) :: dvEdge

      integer, dimension(nCells+1) :: nEdgesOnCell
      integer, dimension(2,nEdges+1) :: cellsOnEdge
      integer, dimension(maxEdges,nCells+1) :: edgesOnCell
      real (kind=RKIND), dimension(maxEdges,nCells+1) :: edgesOnCell_sign

      integer, intent(in) :: small_step
      real (kind=RKIND), intent(in) :: dts, epssm, cf1, cf2, cf3
      real (kind=RKIND), dimension(nVertLevels) :: ts, rs
  
      !
      ! Local variables
      !
      integer :: cell1, cell2, iEdge, iCell, i, k
      real (kind=RKIND) :: c2, rcv, rtheta_pp_tmp
      real (kind=RKIND) :: pgrad, flux, resm, rdts
      real :: T1, T2
      real, intent(inout) :: T_e2e
      integer, intent(in) :: use_gpu

      !print *, "small_step,nCells,nEdges,nCellsSolve,cellStart,cellEnd,vertexStart,vertexEnd,edgeStart,edgeEnd,cellSolveStart,cellSolveEnd,vertexSolveStart,vertexSolveEnd,edgeSolveStart,edgeSolveEnd,dts,epssm,cf1,cf2,cf3"
      !print *, small_step, ",", nCells, ",", nEdges, ",", nCellsSolve, ",", cellStart, ",", cellEnd, ",", vertexStart, ",", vertexEnd, ",", edgeStart, ",", edgeEnd, ",", cellSolveStart, ",", cellSolveEnd, ",", vertexSolveStart, ",", vertexSolveEnd, ",", edgeSolveStart, ",", edgeSolveEnd, ",", dts, ",", epssm, ",", cf1, ",", cf2, ",", cf3

      resm = (1.0 - epssm) / (1.0 + epssm)
      rdts = 1./dts
  
      call random_2D(rho_zz)
      call random_2D(theta_m)
      call random_2D(ru_p)
      call random_2D(rw_p)
      call random_2D(rtheta_pp)
      call random_2D(rtheta_pp_old)
      call random_2D(zz)
      call random_2D(exner)
      call random_2D(cqu)
      call random_2D(ruAvg)
      call random_2D(wwAvg)
      call random_2D(rho_pp)
      call random_2D(cofwt)
      call random_2D(coftz)
      call random_2D(zxu)
      call random_2D(a_tri)
      call random_2D(alpha_tri)
      call random_2D(gamma_tri)
      call random_2D(dss)
      call random_2D(tend_ru)
      call random_2D(tend_rho)
      call random_2D(tend_rt)
      call random_2D(tend_rw)
      call random_2D(zgrid)
      call random_2D(cofwr)
      call random_2D(cofwz)
      call random_2D(w)
      call random_2D(ru)
      call random_2D(ru_save)
      call random_2D(rw)
      call random_2D(rw_save)

      call random_1D(fzm)
      call random_1D(fzp)
      call random_1D(rdzw)
      call random_1D(dcEdge)
      call random_1D(invDcEdge)
      call random_1D(invAreaCell)
      call random_1D(cofrz)
      call random_1D(dvEdge)

      call random_1D_int(nEdgesOnCell, NMAX)
      nEdgesOnCell(:) = max(1, mod(nEdgesOnCell(:), maxEdges))
      call random_2D_int(cellsOnEdge, NMAX)
      cellsOnEdge(:,:) = max(1, mod(cellsOnEdge(:,:), nCells))
      call random_2D_int(edgesOnCell, NMAX)
      edgesOnCell(:,:) = max(1, mod(edgesOnCell(:,:), nEdges))

      call random_2D(edgesOnCell_sign)
!$acc data present(rtheta_pp, rtheta_pp_old, ru_p, ruavg, rho_pp, &
!$acc rw_p, wwavg, &
!$acc zz, cellsonedge, cqu, dcedge, exner, invdcedge, &
!$acc tend_ru, zxu, tend_rho, a_tri, alpha_tri, cofrz, &
!$acc coftz, cofwr, cofwt, cofwz, dss, dvedge, edgesoncell, edgesoncell_sign, &
!$acc fzm, fzp, gamma_tri, invareacell, nedgesoncell, rdzw, rho_zz, rw, &
!$acc rw_save, tend_rho, tend_rt, tend_rw, theta_m, w)

      if(small_step /= 1) then  !  not needed on first small step 

!!!kernel 2: atm_time_integration_atm_advance_acoustic_step_work_2660_gpu
!$acc parallel vector_length(32)
!$acc loop gang
        do iEdge=edgeStart,edgeEnd ! MGD do we really just need edges touching owned cells?

           cell1 = cellsOnEdge(1,iEdge)
           cell2 = cellsOnEdge(2,iEdge)

           ! update edges for block-owned cells
           if (cell1 <= nCellsSolve .or. cell2 <= nCellsSolve ) then
!DIR$ IVDEP
!$acc loop vector
              do k=1,nVertLevels
                 pgrad = ((rtheta_pp(k,cell2)-rtheta_pp(k,cell1))*invDcEdge(iEdge) )/(.5*(zz(k,cell2)+zz(k,cell1)))
                 pgrad = cqu(k,iEdge)*0.5*c2*(exner(k,cell1)+exner(k,cell2))*pgrad
                 pgrad = pgrad + 0.5*zxu(k,iEdge)*gravity*(rho_pp(k,cell1)+rho_pp(k,cell2))
                 ru_p(k,iEdge) = ru_p(k,iEdge) + dts*(tend_ru(k,iEdge) - pgrad)
              end do

              ! accumulate ru_p for use later in scalar transport
!DIR$ IVDEP
!$acc loop vector

              do k=1,nVertLevels
                 ruAvg(k,iEdge) = ruAvg(k,iEdge) + ru_p(k,iEdge)
              end do

           end if ! end test for block-owned cells

        end do ! end loop over edges
!$acc end parallel
        end if

        if(small_step ==1) then
!      else !  this is all that us needed for ru_p update for first acoustic step in RK substep
!$acc parallel vector_length(32)
!$acc loop gang
        do iEdge=edgeStart,edgeEnd ! MGD do we really just need edges touching owned cells?

           cell1 = cellsOnEdge(1,iEdge)
           cell2 = cellsOnEdge(2,iEdge)

           ! update edges for block-owned cells
           if (cell1 <= nCellsSolve .or. cell2 <= nCellsSolve ) then
!DIR$ IVDEP
!$acc loop vector
              do k=1,nVertLevels
                 ru_p(k,iEdge) = dts*tend_ru(k,iEdge)
              end do
!DIR$ IVDEP
!$acc loop vector
              do k=1,nVertLevels
!!                 ruAvg(k,iEdge) = ruAvg(k,iEdge) + ru_p(k,iEdge)
                 ruAvg(k,iEdge) = ru_p(k,iEdge)                 
              end do

           end if ! end test for block-owned cells
        end do ! end loop over edges
!$acc end parallel
      end if ! test for first acoustic step

      if (small_step == 1) then  ! initialize here on first small timestep.
!$acc parallel vector_length(32)
!$acc loop gang vector
         do iCell=cellStart,cellEnd
            rtheta_pp_old(1:nVertLevels,iCell) = 0.0
         end do
!$acc end parallel
      else
!$acc parallel vector_length(32)
!$acc loop gang vector
        do iCell=cellStart,cellEnd
           rtheta_pp_old(1:nVertLevels,iCell) = rtheta_pp(1:nVertLevels,iCell)
        end do
!$acc end parallel
      end if


!!!OMP BARRIER -- not needed, since rtheta_pp_old not used below when small_step == 1
!!!kernel 1: atm_time_integration_atm_advance_acoustic_step_work_2739_gpu
      call cpu_time(T1)
!$omp target if(use_gpu)
!$acc parallel vector_length(32)
!$acc loop gang private(ts, rs)
      do iCell=cellSolveStart,cellSolveEnd  ! loop over all owned cells to solve
!!$acc cache(ts)
!!$acc cache(rs)

         ts(:) = 0.0
         rs(:) = 0.0

         if(small_step == 1) then  ! initialize here on first small timestep.
            wwAvg(1:nVertLevels+1,iCell) = 0.0            
            rho_pp(1:nVertLevels,iCell) = 0.0            
            rtheta_pp(1:nVertLevels,iCell) = 0.0            
!MGD moved to loop above over all cells
!            rtheta_pp_old(1:nVertLevels,iCell) = 0.0
            rw_p(:,iCell) = 0.0
        end if

!$acc loop seq
         do i=1,nEdgesOnCell(iCell) 
            iEdge = edgesOnCell(i,iCell)
            cell1 = cellsOnEdge(1,iEdge)
            cell2 = cellsOnEdge(2,iEdge)
!DIR$ IVDEP
!$acc loop vector

            do k=1,nVertLevels
               flux = edgesOnCell_sign(i,iCell)*dts*dvEdge(iEdge)*ru_p(k,iEdge) * invAreaCell(iCell)
               rs(k) = rs(k)-flux
               ts(k) = ts(k)-flux*0.5*(theta_m(k,cell2)+theta_m(k,cell1))
            end do
         end do
      ! vertically implicit acoustic and gravity wave integration.
      ! this follows Klemp et al MWR 2007, with the addition of an implicit Rayleigh damping of w
      ! serves as a gravity-wave absorbing layer, from Klemp et al 2008.

!DIR$ IVDEP
!$acc loop vector

         do k=1, nVertLevels
            rs(k) = rho_pp(k,iCell) + dts*tend_rho(k,iCell) + rs(k)                  &
                            - cofrz(k)*resm*(rw_p(k+1,iCell)-rw_p(k,iCell)) 
            ts(k) = rtheta_pp(k,iCell) + dts*tend_rt(k,iCell) + ts(k)                &
                               - resm*rdzw(k)*( coftz(k+1,iCell)*rw_p(k+1,iCell)     &
                                               -coftz(k,iCell)*rw_p(k,iCell))
         end do

!DIR$ IVDEP
!$acc loop vector

         do k=2, nVertLevels
            wwavg(k,iCell) = wwavg(k,iCell) + 0.5*(1.0-epssm)*rw_p(k,iCell)
         end do

!DIR$ IVDEP
!$acc loop vector

         do k=2, nVertLevels
            rw_p(k,iCell) = rw_p(k,iCell) +  dts*tend_rw(k,iCell)                       &
                       - cofwz(k,iCell)*((zz(k  ,iCell)*ts(k)                           &
                                     -zz(k-1,iCell)*ts(k-1))                            &
                               +resm*(zz(k  ,iCell)*rtheta_pp(k  ,iCell)                &
                                     -zz(k-1,iCell)*rtheta_pp(k-1,iCell)))              &
                       - cofwr(k,iCell)*((rs(k)+rs(k-1))                                &
                               +resm*(rho_pp(k,iCell)+rho_pp(k-1,iCell)))               &
                       + cofwt(k  ,iCell)*(ts(k)+resm*rtheta_pp(k  ,iCell))           &
                       + cofwt(k-1,iCell)*(ts(k-1)+resm*rtheta_pp(k-1,iCell))
         end do

         ! tridiagonal solve sweeping up and then down the column

!MGD VECTOR DEPENDENCE
!$acc loop seq
         do k=2,nVertLevels
            rw_p(k,iCell) = (rw_p(k,iCell)-a_tri(k,iCell)*rw_p(k-1,iCell))*alpha_tri(k,iCell)
         end do

!MGD VECTOR DEPENDENCE
!$acc loop seq
         do k=nVertLevels,1,-1
            rw_p(k,iCell) = rw_p(k,iCell) - gamma_tri(k,iCell)*rw_p(k+1,iCell)     
         end do

         ! the implicit Rayleigh damping on w (gravity-wave absorbing) 
!DIR$ IVDEP
!$acc loop vector

         do k=2,nVertLevels
            rw_p(k,iCell) = (rw_p(k,iCell) + (rw_save(k  ,iCell) - rw(k  ,iCell)) -dts*dss(k,iCell)*               &
                        (fzm(k)*zz (k,iCell)+fzp(k)*zz (k-1,iCell))        &
                        *(fzm(k)*rho_zz(k,iCell)+fzp(k)*rho_zz(k-1,iCell))       &
                                 *w(k,iCell)    )/(1.0+dts*dss(k,iCell)) &
                         - (rw_save(k  ,iCell) - rw(k  ,iCell))
         end do

         ! accumulate (rho*omega)' for use later in scalar transport
!DIR$ IVDEP
!$acc loop vector

         do k=2,nVertLevels
            wwAvg(k,iCell) = wwAvg(k,iCell) + 0.5*(1.0+epssm)*rw_p(k,iCell)
         end do

         ! update rho_pp and theta_pp given updated rw_p
!DIR$ IVDEP
!$acc loop vector

         do k=1,nVertLevels
            rho_pp(k,iCell) = rs(k) - cofrz(k) *(rw_p(k+1,iCell)-rw_p(k  ,iCell))
            rtheta_pp(k,iCell) = ts(k) - rdzw(k)*(coftz(k+1,iCell)*rw_p(k+1,iCell)  &
                               -coftz(k  ,iCell)*rw_p(k  ,iCell))
         end do
      end do !  end of loop over cells
!$omp end target
      call cpu_time(T2)
      T_e2e = T2 - T1
      print *, "kernel1: ", T_e2e
!$acc end parallel
!$acc end data
      open(1, file = 'mpas_test_kernel1.out')  
      write(1, '(F21.6)') wwAvg
      write(1, '(F21.6)') rho_pp
      write(1, '(F21.6)') rtheta_pp
   end subroutine kernel1
end module mpas_ut_func

program test
    use mpas_ut_func
    use m_random

    implicit none

    real (kind=RKIND), parameter :: dts = 75.0
    real (kind=RKIND), parameter :: epssm = 0.1
    real (kind=RKIND), parameter :: cf1 = 2.0
    real (kind=RKIND), parameter :: cf2 = -1.5
    real (kind=RKIND), parameter :: cf3 = -0.5
    integer :: small_step = 6
    integer :: i, iter, use_gpu
    real :: t_sum = 0, t_iter
    character(len=12) :: args

    call get_command_argument(1, args)
    read(args,'(i)') use_gpu

    iter = 1
    call random_init()
    call kernel1(26,10,10969,33274,10238,1,10969,1,22306,1,33274,1,10238,1,20453,1,30696,dts,small_step,epssm,cf1,cf2,cf3, t_iter, use_gpu)
   print *, "jw_k1: ", (t_sum / iter)
   print *, t_iter
end program test
