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

  subroutine calc_current(ppg, dz)
    use structures

    implicit none
    type(s_dft_system) :: system
    type(s_rgrid)   :: mg
    type(s_parallel_info) :: info
    type(s_orbital)            :: psi
    type(s_pp_grid), intent(inout) :: ppg
    integer :: ispin,im,ik,io,nspin,ngrid
    real(8),dimension(3) :: wrk1,wrk2,wrk3,wrk4
    real(8) :: BT(3,3),kAc(3)
    complex(8),intent(in) :: dz
    real(8) :: jx,jy,jz
    real(8) :: cpu_jx,cpu_jy,cpu_jz
    integer :: i,j,k,ia,ilma

    !$omp declare target(calc_current_nonlocal)
    ispin = 1
    info%io_s = 1
    info%ik_s = 1
    info%io_e = 48
    info%ik_e = 64
    mg%is_array(1) = 1
    mg%is_array(2) = 1
    mg%is_array(3) = 1
    mg%ie_array(1) = 20
    mg%ie_array(2) = 37
    mg%ie_array(3) = 52
    im = 1
    allocate(system%rocc(info%io_e,info%ik_e,ispin))
    allocate(system%wtk(info%ik_e))
    allocate(psi%zwf(mg%ie_array(1),mg%ie_array(2),mg%ie_array(3),ispin,info%io_e,info%ik_e,im))
    allocate(ppg%zekr_uV(ppg%Mps(1),ppg%Nlma,info%ik_e))

    do i=1,ppg%Nlma
        ia=ppg%ia_tbl(i)
        ppg%rinv_uvu(i) = real(i)
        do j=1,ppg%Mps(ia)
            ppg%Rxyz(1,j,ia) = 1.0d0
            ppg%Rxyz(2,j,ia) = 1.0d0
            ppg%Rxyz(3,j,ia) = 1.0d0
            ppg%Jxyz(1,j,ia) = mod(j, mg%ie_array(1)) + 1
            ppg%Jxyz(2,j,ia) = mod(j, mg%ie_array(2)) + 1
            ppg%Jxyz(3,j,ia) = mod(j, mg%ie_array(3)) + 1
            do k=1,info%ik_e
                ppg%zekr_uV(j,i,k) = conjg(dz)
            enddo
        enddo
    enddo
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
        system%rocc(io,ik,ispin) = 1.0d0
        system%wtk(ik) = 1.0d0
        do i=1,mg%ie_array(1)
            do j=1,mg%ie_array(2)
                do k=1,mg%ie_array(3)
                    psi%zwf(i,j,k,1,io,ik,1) = conjg(dz)
                enddo
            enddo
        enddo
    enddo
    enddo
    call flush(6)
    jx = 0d0
    jy = 0d0
    jz = 0d0
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
      call calc_current_nonlocal(wrk3,psi%zwf(:,:,:,ispin,io,ik,im),ppg,mg%is_array,mg%ie_array,ik)
      wrk4 = wrk3 * system%rocc(io,ik,ispin) * system%wtk(ik)
      jx = jx + wrk4(1)
      jy = jy + wrk4(2)
      jz = jz + wrk4(3)
    end do
    end do
    cpu_jx = jx
    cpu_jy = jy
    cpu_jz = jz

    jx = 0d0
    jy = 0d0
    jz = 0d0
!$acc kernels copyin(ispin,im) copy(jx,jy,jz)
!$acc loop gang private(ik,io,wrk3,wrk4) reduction(+:jx,jy,jz) collapse(2) independent
#ifdef USE_GPU
!$omp target teams distribute parallel do map(to:ispin,im) map(tofrom:jx,jy,jz) private(ik,io,wrk3,wrk4) reduction(+:jx,jy,jz) collapse(2)
#endif
    do ik=info%ik_s,info%ik_e
    do io=info%io_s,info%io_e
      call calc_current_nonlocal(wrk3,psi%zwf(:,:,:,ispin,io,ik,im),ppg,mg%is_array,mg%ie_array,ik)
      wrk4 = wrk3 * system%rocc(io,ik,ispin) * system%wtk(ik)
      jx = jx + wrk4(1)
      jy = jy + wrk4(2)
      jz = jz + wrk4(3)
    end do
    end do
!$acc end kernels
    if (.not. cpu_jx .eq. jx) print *, "jx:", cpu_jx, jx
    if (.not. cpu_jy .eq. jy) print *, "jy:", cpu_jy, jy
    if (.not. cpu_jz .eq. jz) print *, "jz:", cpu_jz, jz

    return

  end subroutine calc_current

  subroutine calc_current_nonlocal(jw,psi,ppg,is_array,ie_array,ik)
    !$acc routine worker
    use structures
    implicit none
    integer,intent(in)    :: is_array(3),ie_array(3),ik
    complex(8),intent(in) :: psi(is_array(1):ie_array(1),is_array(2):ie_array(2),is_array(3):ie_array(3))
    !complex(8),intent(in) :: psi(:,:,:)
    type(s_pp_grid),intent(in) :: ppg
    real(8)               :: jw(3)
    real(8)               :: jw_1, jw_2, jw_3
    !
    integer    :: ilma,ia,j,ix,iy,iz
    real(8)    :: x,y,z
    complex(8) :: uVpsi,uVpsi_r(3)
    !$omp declare target
    jw = 0d0
  jw_1 = 0d0
  jw_2 = 0d0
  jw_3 = 0d0
    do ilma=1,ppg%Nlma
      ia=ppg%ia_tbl(ilma)
      uVpsi = 0d0
      uVpsi_r(1) = 0d0
      uVpsi_r(2) = 0d0
      uVpsi_r(3) = 0d0
      do j=1,ppg%Mps(ia)
        x = ppg%Rxyz(1,j,ia)
        y = ppg%Rxyz(2,j,ia)
        z = ppg%Rxyz(3,j,ia)
        ix = ppg%Jxyz(1,j,ia)
        iy = ppg%Jxyz(2,j,ia)
        iz = ppg%Jxyz(3,j,ia)
        uVpsi = uVpsi + conjg(ppg%zekr_uV(j,ilma,ik)) * conjg(ppg%zekr_uV(j,ilma,ik))!psi(ix,iy,iz)
        uVpsi_r(1) = uVpsi_r(1) + conjg(ppg%zekr_uV(j,ilma,ik)) * x * psi(ix,iy,iz)
        uVpsi_r(2) = uVpsi_r(2) + conjg(ppg%zekr_uV(j,ilma,ik)) * y * psi(ix,iy,iz)
        uVpsi_r(3) = uVpsi_r(3) + conjg(ppg%zekr_uV(j,ilma,ik)) * z * psi(ix,iy,iz)
      end do
      uVpsi = uVpsi * ppg%rinv_uvu(ilma)
      jw_1 = jw_1 + aimag(conjg(uVpsi_r(1))*uVpsi)
      jw_2 = jw_2 + aimag(conjg(uVpsi_r(2))*uVpsi)
      jw_3 = jw_3 + aimag(conjg(uVpsi_r(3))*uVpsi)
    end do
  jw(1) = jw_1
  jw(2) = jw_2
  jw(3) = jw_3
    return
  end subroutine calc_current_nonlocal
end module

program main
    use density_matrix
    use structures 
    type(s_pp_grid) :: ppg
    complex :: z = (2.0, 3.0)
    complex(8) :: dz = (2.71_8, -3.14_8)
    !complex(8) :: dz = (1.0_8, -1.0_8)

    allocate(ppg%Mps(1))
    ppg%Mps(1) = 1000
    ppg%Nlma = 1
    !allocate(ppg%zekr_uV(ppg%Mps(1),ppg%Nlma,64))
    allocate(ppg%Rxyz(3,ppg%Mps(1),ppg%Nlma))
    allocate(ppg%Jxyz(3,ppg%Mps(1),ppg%Nlma))
    allocate(ppg%rinv_uvu(ppg%Nlma))
    allocate(ppg%ia_tbl(ppg%Nlma))
    ppg%ia_tbl(1) = 1

    dz = dconjg(dz)
    call calc_current(ppg, dz)
end program
