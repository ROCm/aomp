! Copyrieht (c) 2013,  Los Alamos National Security, LLC (LANS)
! and the University Corporation for Atmospheric Research (UCAR).
!
! Unless noted otherwise source code is licensed under the BSD license.
! Additional copyright and license information can be found in the LICENSE file
! distributed with this code, or at http://mpas-dev.github.com/license.html
!

module m_init
   contains

   subroutine init_1D(x)
     real :: x(:)

     x(:) = 0.5
     !write(*,'(F21.18)') x
   end subroutine init_1D

   subroutine init_1D_int(x, max_val)
     integer :: x(:)
     real :: max_val

     x(:) = 0.5 * max_val
     !write(*,'(I5)') x
   end subroutine init_1D_int

   subroutine init_2D(x)
     real:: x(:,:)

     x(:,:) = 0.5
     !write(*,'(F21.18)') x
   end subroutine init_2D

   subroutine init_2D_int(x, max_val)
     integer :: x(:,:)
     real :: max_val

     x(:,:) = 0.5 * max_val
     !write(*,'(I5)') x
   end subroutine init_2D_int
end module m_init


module mpas_ut_func
   integer, parameter :: RKIND  = selected_real_kind(12)
   real, parameter :: pii     = 3.141592653589793   !< Constant: Pi
   real, parameter :: a       = 6371229.0           !< Constant: Spherical Earth radius [m]
   real, parameter :: omega   = 7.29212e-5          !< Constant: Angular rotation rate of the Earth [s-1]
   real, parameter :: gravity = 9.80616             !< Constant: Acceleration due to gravity [m s-2]
   real, parameter :: rgas    = 287.0               !< Constant: Gas constant for dry air [J kg-1 K-1]
   real, parameter :: rv      = 461.6               !< Constant: Gas constant for water vapor [J kg-1 K-1]
   real, parameter :: rvord   = rv*rgas             !
   real, parameter :: cp      = 7.*rgas*2.          !< Constant: Specific heat of dry air at constant pressure [J kg-1 K-1]
   !real, parameter :: rvord   = rv/rgas             !
   !real, parameter :: cp      = 7.*rgas/2.          !< Constant: Specific heat of dry air at constant pressure [J kg-1 K-1]
   real, parameter :: cv      = cp - rgas           !< Constant: Specific heat of dry air at constant volume [J kg-1 K-1]
   real, parameter :: cvpm    = -cv*cp              !
   real, parameter :: prandtl = 1.0                 !< Constant: Prandtl number
   real, parameter :: NMAX    = 10000000.0

   contains

   subroutine kernel1(nVertLevels, maxEdges, nCells, nEdges, nCellsSolve, cellStart, cellEnd, vertexStart, vertexEnd, edgeStart, edgeEnd, &
                                   cellSolveStart, cellSolveEnd, vertexSolveStart, vertexSolveEnd, edgeSolveStart, edgeSolveEnd, &
                                   dts, small_step, epssm, cf1, cf2, cf3, T_e2e, use_gpu&
                                   )
      use m_init
      implicit none

      !
      ! Dummy arguments
      !
      integer, intent(in) :: nVertLevels, maxEdges
      integer, intent(in) :: nCells, nEdges, nCellsSolve
      integer, intent(in) :: cellStart, cellEnd, vertexStart, vertexEnd, edgeStart, edgeEnd
      integer, intent(in) :: cellSolveStart, cellSolveEnd, vertexSolveStart, vertexSolveEnd, edgeSolveStart, edgeSolveEnd

      real, dimension(nVertLevels,nCells+1) :: rho_zz
      real, dimension(nVertLevels,nCells+1) :: theta_m
      real, dimension(nVertLevels,nEdges+1) :: ru_p
      real, dimension(nVertLevels+1,nCells+1) :: rw_p
      real, dimension(nVertLevels,nCells+1) :: rtheta_pp

      real, dimension(nVertLevels,nCells+1) :: rtheta_pp_old
      real, dimension(nVertLevels,nCells+1) :: zz
      real, dimension(nVertLevels,nCells+1) :: exner
      real, dimension(nVertLevels,nEdges+1) :: cqu
      real, dimension(nVertLevels,nEdges+1) :: ruAvg
      real, dimension(nVertLevels+1,nCells+1) :: wwAvg
      real, dimension(nVertLevels,nCells+1) :: rho_pp
      real, dimension(nVertLevels,nCells+1) :: cofwt
      real, dimension(nVertLevels+1,nCells+1) :: coftz
      real, dimension(nVertLevels,nEdges+1) :: zxu

      real, dimension(nVertLevels,nCells+1) :: a_tri
      real, dimension(nVertLevels,nCells+1) :: alpha_tri
      real, dimension(nVertLevels,nCells+1) :: gamma_tri
      real, dimension(nVertLevels,nCells+1) :: dss
      real, dimension(nVertLevels,nEdges+1) :: tend_ru
      real, dimension(nVertLevels,nCells+1) :: tend_rho
      real, dimension(nVertLevels,nCells+1) :: tend_rt
      real, dimension(nVertLevels+1,nCells+1) :: tend_rw
      real, dimension(nVertLevels+1,nCells+1) :: zgrid
      real, dimension(nVertLevels,nCells+1) :: cofwr
      real, dimension(nVertLevels,nCells+1) :: cofwz

      real, dimension(nVertLevels+1,nCells+1) :: w
      real, dimension(nVertLevels,nEdges+1) :: ru
      real, dimension(nVertLevels,nEdges+1) :: ru_save
      real, dimension(nVertLevels+1,nCells+1) :: rw
      real, dimension(nVertLevels+1,nCells+1) :: rw_save

      real, dimension(nVertLevels) :: fzm
      real, dimension(nVertLevels) :: fzp
      real, dimension(nVertLevels) :: rdzw
      real, dimension(nEdges+1) :: dcEdge
      real, dimension(nEdges+1) :: invDcEdge
      real, dimension(nCells+1) :: invAreaCell
      real, dimension(nVertLevels) :: cofrz
      real, dimension(nEdges+1) :: dvEdge

      integer, dimension(nCells+1) :: nEdgesOnCell
      integer, dimension(2,nEdges+1) :: cellsOnEdge
      integer, dimension(maxEdges,nCells+1) :: edgesOnCell
      real, dimension(maxEdges,nCells+1) :: edgesOnCell_sign

      integer, intent(in) :: small_step
      real, intent(in) :: dts, epssm, cf1, cf2, cf3
      real, dimension(nVertLevels) :: ts, rs
  
      !
      ! Local variables
      !
      integer :: cell1, cell2, iEdge, iCell, i, k
      real :: c2, rcv, rtheta_pp_tmp
      real :: pgrad, flux, resm, rdts
      real, intent(inout) :: T_e2e
      integer, intent(in) :: use_gpu

      !print *, "small_step,nCells,nEdges,nCellsSolve,cellStart,cellEnd,vertexStart,vertexEnd,edgeStart,edgeEnd,cellSolveStart,cellSolveEnd,vertexSolveStart,vertexSolveEnd,edgeSolveStart,edgeSolveEnd,dts,epssm,cf1,cf2,cf3"
      !print *, small_step, ",", nCells, ",", nEdges, ",", nCellsSolve, ",", cellStart, ",", cellEnd, ",", vertexStart, ",", vertexEnd, ",", edgeStart, ",", edgeEnd, ",", cellSolveStart, ",", cellSolveEnd, ",", vertexSolveStart, ",", vertexSolveEnd, ",", edgeSolveStart, ",", edgeSolveEnd, ",", dts, ",", epssm, ",", cf1, ",", cf2, ",", cf3

      resm = (1.0 - epssm) / (1.0 + epssm)
      rdts = 1./dts
      c2 = 0.5
  
      call init_2D(rho_zz)
      call init_2D(theta_m)
      call init_2D(ru_p)
      call init_2D(rw_p)
      call init_2D(rtheta_pp)
      call init_2D(rtheta_pp_old)
      call init_2D(zz)
      call init_2D(exner)
      call init_2D(cqu)
      call init_2D(ruAvg)
      call init_2D(wwAvg)
      call init_2D(rho_pp)
      call init_2D(cofwt)
      call init_2D(coftz)
      call init_2D(zxu)
      call init_2D(a_tri)
      call init_2D(alpha_tri)
      call init_2D(gamma_tri)
      call init_2D(dss)
      call init_2D(tend_ru)
      call init_2D(tend_rho)
      call init_2D(tend_rt)
      call init_2D(tend_rw)
      call init_2D(zgrid)
      call init_2D(cofwr)
      call init_2D(cofwz)
      call init_2D(w)
      call init_2D(ru)
      call init_2D(ru_save)
      call init_2D(rw)
      call init_2D(rw_save)

      call init_1D(fzm)
      call init_1D(fzp)
      call init_1D(rdzw)
      call init_1D(dcEdge)
      call init_1D(invDcEdge)
      call init_1D(invAreaCell)
      call init_1D(cofrz)
      call init_1D(dvEdge)

      call init_1D_int(nEdgesOnCell, NMAX)
      nEdgesOnCell(:) = max(1, mod(nEdgesOnCell(:), maxEdges))
      call init_2D_int(cellsOnEdge, NMAX)
      cellsOnEdge(:,:) = max(1, mod(cellsOnEdge(:,:), nCells))
      call init_2D_int(edgesOnCell, NMAX)
      edgesOnCell(:,:) = max(1, mod(edgesOnCell(:,:), nEdges))

      call init_2D(edgesOnCell_sign)
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
#ifdef ONE_TEAM
!$omp target &
#elif M_TEAMS
!$omp target teams &
#elif M_TEAMS_N_THREADS
!$omp target teams distribute &
#endif
#ifdef PRIV_TS_RS
!$omp private(ts, rs) &
#endif
!$omp
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
#ifdef ONE_TEAM
!$omp end target
#elif M_TEAMS
!$omp end target teams
#elif CPU_M_TEAMS
!$omp end teams
#endif
!$acc end parallel
!$acc end data
      open(1, file = 'mpas_test_kernel1.out')  
      write(1, '(F21.6)') wwAvg
      write(1, '(F21.6)') rho_pp
      write(1, '(F21.6)') rtheta_pp
      !print *, wwAvg, rho_pp, rtheta_pp
   end subroutine kernel1
end module mpas_ut_func

program test
    use mpas_ut_func
    use m_init

    implicit none

    real, parameter :: dts = 75.0
    real, parameter :: epssm = 0.1
    real, parameter :: cf1 = 2.0
    real, parameter :: cf2 = -1.5
    real, parameter :: cf3 = -0.5
    integer :: small_step = 6
    integer :: i, iter, use_gpu
    real :: t_sum = 0, t_iter
    character(len=12) :: args

    call get_command_argument(1, args)
    read(args,'(i)') use_gpu

    iter = 1
    call kernel1(26,10,10969,33274,10238,1,10969,1,22306,1,33274,1,10238,1,20453,1,30696,dts,small_step,epssm,cf1,cf2,cf3, t_iter, use_gpu)
    print *, "PASS"
    return
end program test
