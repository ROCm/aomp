!
!  Copyright 2019-2020 SALMON developers
!
!  Licensed under the Apache License, Version 2.0 (the "License");
!  you may not use this file except in compliance with the License.
!  You may obtain a copy of the License at
!
!      http://www.apache.org/licenses/LICENSE-2.0
!
!  Unless required by applicable law or agreed to in writing, software
!  distributed under the License is distributed on an "AS IS" BASIS,
!  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
!  See the License for the specific language governing permissions and
!  limitations under the License.
!
!--------10--------20--------30--------40--------50--------60--------70--------80--------90--------100-------110-------120-------130


module density_matrix
  implicit none
  integer,private,parameter :: Nd = 4

contains

  subroutine stencil_current(is_array,ie_array,is,ie,idx,idy,idz,nabt,kAc,psi,j1,j2)
    !$acc routine worker
    !$omp declare target
    integer   ,intent(in) :: is_array(3),ie_array(3),is(3),ie(3) &
                            ,idx(is(1)-Nd:ie(1)+Nd),idy(is(2)-Nd:ie(2)+Nd),idz(is(3)-Nd:ie(3)+Nd)
    real(8)   ,intent(in) :: nabt(Nd,3),kAc(3)
    complex(8),intent(in) :: psi(is_array(1):ie_array(1),is_array(2):ie_array(2),is_array(3):ie_array(3))
    real(8)               :: j1(3),j2(3)
    !
    integer :: ix,iy,iz
    real(8) :: rtmp
    complex(8) :: cpsi,xtmp,ytmp,ztmp
    rtmp = 0d0
    xtmp = 0d0
    ytmp = 0d0
    ztmp = 0d0
!$acc loop vector collapse(2) private(iz,iy,ix,cpsi) reduction(+:rtmp,xtmp,ytmp,ztmp)
!$omp parallel do collapse(2) private(iz,iy,ix,cpsi) &
#ifdef USE_REDUCE
!$omp& reduction(+:rtmp,xtmp,ytmp,ztmp) &
#endif
!$omp&

    do iz=is(3),ie(3)
    do iy=is(2),ie(2)

!OCL swp
    do ix=is(1),ie(1)
      rtmp = rtmp + abs(psi(ix,iy,iz))**2
    end do

!OCL swp
    do ix=is(1),ie(1)
      cpsi = conjg(psi(ix,iy,iz))
      xtmp = xtmp + nabt(1,1) * cpsi * psi(idx(ix+1),iy,iz) &
                  + nabt(2,1) * cpsi * psi(idx(ix+2),iy,iz) &
                  + nabt(3,1) * cpsi * psi(idx(ix+3),iy,iz) &
                  + nabt(4,1) * cpsi * psi(idx(ix+4),iy,iz)
    end do

!OCL swp
    do ix=is(1),ie(1)
      cpsi = conjg(psi(ix,iy,iz))
      ytmp = ytmp + nabt(1,2) * cpsi * psi(ix,idy(iy+1),iz) &
                  + nabt(2,2) * cpsi * psi(ix,idy(iy+2),iz) &
                  + nabt(3,2) * cpsi * psi(ix,idy(iy+3),iz) &
                  + nabt(4,2) * cpsi * psi(ix,idy(iy+4),iz)
    end do

!OCL swp
    do ix=is(1),ie(1)
      cpsi = conjg(psi(ix,iy,iz))
      ztmp = ztmp + nabt(1,3) * cpsi * psi(ix,iy,idz(iz+1)) &
                  + nabt(2,3) * cpsi * psi(ix,iy,idz(iz+2)) &
                  + nabt(3,3) * cpsi * psi(ix,iy,idz(iz+3)) &
                  + nabt(4,3) * cpsi * psi(ix,iy,idz(iz+4))
    end do

    end do
    end do
    j1 = kAc(:) * rtmp
    j2(1) = aimag(xtmp * 2d0)
    j2(2) = aimag(ytmp * 2d0)
    j2(3) = aimag(ztmp * 2d0)
    return
  end subroutine stencil_current
end module

program main
    print *, "Hello"
end program
