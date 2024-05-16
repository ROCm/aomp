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

module nonlocal_potential
  implicit none

! WARNING: We must not call these except for hpsi routine.

contains

!-----------------------------------------------------------------------------------------------------------------------------------

subroutine zpseudo_gpu(info,htpsi,mg) !(tpsi,htpsi,info,nspin,ppg)
  use structures
  implicit none
  integer :: nspin = 1
  type(s_parallel_info) :: info
  type(s_pp_grid) :: ppg
  type(s_orbital) :: tpsi
  type(s_orbital) :: htpsi
  type(s_rgrid)   :: mg
  !
  integer :: ispin,io,ik,im,im_s,im_e,ik_s,ik_e,io_s,io_e,i
  integer :: ilma,ia,j,ix,iy,iz,Nlma,ilocal,vi,my_nlma,k
  complex(8) :: uVpsi,wrk
  complex(8),allocatable :: uVpsibox (:,:,:,:,:)
  complex(8),allocatable :: uVpsibox2(:,:,:,:,:)
  complex(8) :: IMAGINARY_UNIT = (0, 1)
  complex(8) :: dz = (2.71_8, -3.14_8)
  !complex(8) :: dz = (1.0_8, -1.0_8)

  info%im_s = 1
  info%im_e = 1
  info%ik_s = 1
  info%ik_e = 64
  info%io_s = 1
  info%io_e = 48
  mg%is_array(1) = 1
  mg%is_array(2) = 1
  mg%is_array(3) = 1
  mg%ie_array(1) = 20
  mg%ie_array(2) = 37
  mg%ie_array(3) = 52
  im_s = info%im_s
  im_e = info%im_e
  ik_s = info%ik_s
  ik_e = info%ik_e
  io_s = info%io_s
  io_e = info%io_e
  nspin = 1
  Nlma = 1
  ppg%Nlma = Nlma
  ppg%ilocal_nlma = Nlma
  ppg%max_vi = 2
  allocate(ppg%k2ilma(0:ppg%max_vi-1,1000))
  allocate(ppg%k2j(0:ppg%max_vi-1,1000))
  allocate(ppg%v2nlma(0:ppg%max_vi-1))
  allocate(ppg%v2j(3,0:ppg%max_vi-1))

  do vi=0,ppg%max_vi-1
    ppg%v2nlma(vi) = 1000
    my_nlma = ppg%v2nlma(vi)
    do k=1,my_nlma
      ppg%k2ilma(vi,k) = 1
      ppg%k2j(vi,k) = mod(k, info%ik_e) + 1
      ppg%v2j(1,vi) = mod(k, mg%ie_array(1))
      ppg%v2j(2,vi) = mod(k, mg%ie_array(2))
      ppg%v2j(3,vi) = mod(k, mg%ie_array(3))
    enddo
  enddo

  call flush(6)
  allocate(ppg%ia_tbl(Nlma))
  allocate(ppg%mps(Nlma))
  ppg%mps(1)=1000
  allocate(ppg%rinv_uvu(Nlma))
  allocate(ppg%Jxyz(3,ppg%Mps(1),ppg%Nlma))
  allocate(ppg%zekr_uV(ppg%Mps(1),ppg%Nlma,info%ik_e))
  allocate(tpsi%zwf(mg%ie_array(1),mg%ie_array(2),mg%ie_array(3),nspin,info%io_e,info%ik_e,im_e))
  allocate(htpsi%zwf(mg%ie_array(1),mg%ie_array(2),mg%ie_array(3),nspin,info%io_e,info%ik_e,im_e))
  allocate(ppg%uVpsibox(Nlma,nspin,io_e,ik_e,im_e))
  allocate(uVpsibox2(nspin,io_e,ik_e,im_e,Nlma))

  do i=1,ppg%Nlma
      ppg%ia_tbl(i)=i
      ia=ppg%ia_tbl(i)
      ppg%rinv_uvu(i) = real(i)
      do j=1,ppg%Mps(ia)
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
      ppg%uVpsibox(1,1,io,ik,1) = conjg(dz)
      do i=1,mg%ie_array(1)
          do j=1,mg%ie_array(2)
              do k=1,mg%ie_array(3)
                  tpsi%zwf(i,j,k,1,io,ik,1) = conjg(dz)
                  htpsi%zwf(i,j,k,1,io,ik,1) = 0.0
              enddo
          enddo
      enddo
  enddo
  enddo

!$acc kernels present(ppg,tpsi,htpsi)
!$acc loop private(ilocal,ilma,ia,uvpsi,vi,my_nlma,k,j,ix,iy,iz,wrk) collapse(4) gang
#ifdef USE_GPU
!$omp target teams distribute parallel do private(ilma,ia,uvpsi,vi,my_nlma,k,j,ix,iy,iz,wrk) collapse(4)
#endif
    do im=im_s,im_e
    do ik=ik_s,ik_e
    do io=io_s,io_e
    do ispin=1,Nspin

!$acc loop gang independent
      do ilma=1,Nlma
        ia = ppg%ia_tbl(ilma)
        uVpsi = 0.d0
!$acc loop vector reduction(+:uVpsi)
        do j=1,ppg%mps(ia)
          ix = ppg%jxyz(1,j,ia)
          iy = ppg%jxyz(2,j,ia)
          iz = ppg%jxyz(3,j,ia)
          uVpsi = uVpsi + conjg(ppg%zekr_uV(j,ilma,ik)) * tpsi%zwf(ix,iy,iz,ispin,io,ik,im)
        end do
        ppg%uVpsibox(ilma,ispin,io,ik,im) = uVpsi * ppg%rinv_uvu(ilma)
      end do

!$acc loop gang vector independent
      do vi=0,ppg%max_vi-1
        my_nlma = ppg%v2nlma(vi)
        if (my_nlma < 1) cycle

        wrk = 0d0
!$acc loop seq
        do k=1,my_nlma
          ilma = ppg%k2ilma(vi,k)
          j    = ppg%k2j(vi,k)
          wrk  = wrk + ppg%uVpsibox(ilma,ispin,io,ik,im) * ppg%zekr_uV(j,ilma,ik)
        end do
        ix = ppg%v2j(1,vi)
        iy = ppg%v2j(2,vi)
        iz = ppg%v2j(3,vi)
        htpsi%zwf(ix,iy,iz,ispin,io,ik,im) = htpsi%zwf(ix,iy,iz,ispin,io,ik,im) + wrk
      end do
    end do
    end do
    end do
    end do
!$acc end kernels
  return
end subroutine zpseudo_gpu

subroutine zpseudo(info,htpsi,mg) !(tpsi,htpsi,info,nspin,ppg)
  use structures
  implicit none
  integer :: nspin = 1
  type(s_parallel_info) :: info
  type(s_pp_grid) :: ppg
  type(s_orbital) :: tpsi
  type(s_orbital) :: htpsi
  type(s_rgrid)   :: mg
  !
  integer :: ispin,io,ik,im,im_s,im_e,ik_s,ik_e,io_s,io_e,i
  integer :: ilma,ia,j,ix,iy,iz,Nlma,ilocal,vi,my_nlma,k
  complex(8) :: uVpsi,wrk
  complex(8),allocatable :: uVpsibox (:,:,:,:,:)
  complex(8),allocatable :: uVpsibox2(:,:,:,:,:)
  complex(8) :: IMAGINARY_UNIT = (0, 1)
  complex(8) :: dz = (2.71_8, -3.14_8)
  !complex(8) :: dz = (1.0_8, -1.0_8)

  info%im_s = 1
  info%im_e = 1
  info%ik_s = 1
  info%ik_e = 64
  info%io_s = 1
  info%io_e = 48
  mg%is_array(1) = 1
  mg%is_array(2) = 1
  mg%is_array(3) = 1
  mg%ie_array(1) = 20
  mg%ie_array(2) = 37
  mg%ie_array(3) = 52
  im_s = info%im_s
  im_e = info%im_e
  ik_s = info%ik_s
  ik_e = info%ik_e
  io_s = info%io_s
  io_e = info%io_e
  nspin = 1
  Nlma = 1
  ppg%Nlma = Nlma
  ppg%ilocal_nlma = Nlma
  ppg%max_vi = 2
  allocate(ppg%k2ilma(0:ppg%max_vi-1,1000))
  allocate(ppg%k2j(0:ppg%max_vi-1,1000))
  allocate(ppg%v2nlma(0:ppg%max_vi-1))
  allocate(ppg%v2j(3,0:ppg%max_vi-1))

  do vi=0,ppg%max_vi-1
    ppg%v2nlma(vi) = 1000
    my_nlma = ppg%v2nlma(vi)
    do k=1,my_nlma
      ppg%k2ilma(vi,k) = 1
      ppg%k2j(vi,k) = mod(k, info%ik_e) + 1
      ppg%v2j(1,vi) = mod(k, mg%ie_array(1))
      ppg%v2j(2,vi) = mod(k, mg%ie_array(2))
      ppg%v2j(3,vi) = mod(k, mg%ie_array(3))
    enddo
  enddo

  call flush(6)
  allocate(ppg%ia_tbl(Nlma))
  allocate(ppg%mps(Nlma))
  ppg%mps(1)=1000
  allocate(ppg%rinv_uvu(Nlma))
  allocate(ppg%Jxyz(3,ppg%Mps(1),ppg%Nlma))
  allocate(ppg%zekr_uV(ppg%Mps(1),ppg%Nlma,info%ik_e))
  allocate(tpsi%zwf(mg%ie_array(1),mg%ie_array(2),mg%ie_array(3),nspin,info%io_e,info%ik_e,im_e))
  allocate(htpsi%zwf(mg%ie_array(1),mg%ie_array(2),mg%ie_array(3),nspin,info%io_e,info%ik_e,im_e))
  allocate(ppg%uVpsibox(Nlma,nspin,io_e,ik_e,im_e))
  allocate(uVpsibox2(nspin,io_e,ik_e,im_e,Nlma))

  do i=1,ppg%Nlma
      ppg%ia_tbl(i)=i
      ia=ppg%ia_tbl(i)
      ppg%rinv_uvu(i) = real(i)
      do j=1,ppg%Mps(ia)
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
      ppg%uVpsibox(1,1,io,ik,1) = conjg(dz)
      do i=1,mg%ie_array(1)
          do j=1,mg%ie_array(2)
              do k=1,mg%ie_array(3)
                  tpsi%zwf(i,j,k,1,io,ik,1) = conjg(dz)
                  htpsi%zwf(i,j,k,1,io,ik,1) = 0.0
              enddo
          enddo
      enddo
  enddo
  enddo

!!$omp target teams distribute parallel do private(ilocal,ilma,ia,uvpsi,vi,my_nlma,k,j,ix,iy,iz,wrk) collapse(4)
!$acc kernels present(ppg,tpsi,htpsi)
!$acc loop private(ilocal,ilma,ia,uvpsi,vi,my_nlma,k,j,ix,iy,iz,wrk) collapse(4) gang
    do im=im_s,im_e
    do ik=ik_s,ik_e
    do io=io_s,io_e
    do ispin=1,Nspin

!$acc loop gang independent
      do ilma=1,Nlma
        ia = ppg%ia_tbl(ilma)
        uVpsi = 0.d0
!$acc loop vector reduction(+:uVpsi)
        do j=1,ppg%mps(ia)
          ix = ppg%jxyz(1,j,ia)
          iy = ppg%jxyz(2,j,ia)
          iz = ppg%jxyz(3,j,ia)
          uVpsi = uVpsi + conjg(ppg%zekr_uV(j,ilma,ik)) * tpsi%zwf(ix,iy,iz,ispin,io,ik,im)
        end do
        ppg%uVpsibox(ilma,ispin,io,ik,im) = uVpsi * ppg%rinv_uvu(ilma)
      end do

!$acc loop gang vector independent
      do vi=0,ppg%max_vi-1
        my_nlma = ppg%v2nlma(vi)
        if (my_nlma < 1) cycle

        wrk = 0d0
!$acc loop seq
        do k=1,my_nlma
          ilma = ppg%k2ilma(vi,k)
          j    = ppg%k2j(vi,k)
          wrk  = wrk + ppg%uVpsibox(ilma,ispin,io,ik,im) * ppg%zekr_uV(j,ilma,ik)
        end do
        ix = ppg%v2j(1,vi)
        iy = ppg%v2j(2,vi)
        iz = ppg%v2j(3,vi)
        htpsi%zwf(ix,iy,iz,ispin,io,ik,im) = htpsi%zwf(ix,iy,iz,ispin,io,ik,im) + wrk
      end do
    end do
    end do
    end do
    end do
!$acc end kernels
  return
end subroutine zpseudo

end module nonlocal_potential

program main
  use nonlocal_potential
  use structures
  type(s_parallel_info) :: info
  type(s_orbital) :: htpsi,htpsi_gpu
  type(s_rgrid)   :: mg
  integer :: cnt = 0

  call zpseudo(info,htpsi,mg)
  call zpseudo_gpu(info,htpsi_gpu,mg)

  do ik=info%ik_s,info%ik_e
  do io=info%io_s,info%io_e
      do i=1,mg%ie_array(1)
          do j=1,mg%ie_array(2)
              do k=1,mg%ie_array(3)
                  if (.not. htpsi%zwf(i,j,k,1,io,ik,1) .eq. htpsi_gpu%zwf(i,j,k,1,io,ik,1) .and. cnt .lt. 20 .and. abs(htpsi%zwf(i,j,k,1,io,ik,1)) .gt. 0) then
                      print *, htpsi%zwf(i,j,k,1,io,ik,1), htpsi_gpu%zwf(i,j,k,1,io,ik,1)
                      cnt = cnt + 1
                  end if
              enddo
          enddo
      enddo
  enddo
  enddo
end program
