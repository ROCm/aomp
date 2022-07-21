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
module func
    implicit none
    type s_scalar
        real(8),allocatable :: f(:,:,:) ! f(x,y,z)
    end type s_scalar
    contains

    subroutine zstencil_typical(io_s,io_e,Nspin,is_array,ie_array,is,ie,idx,idy,idz,igs,ige &
                                   ,tpsi,htpsi,V_local,lap0,lapt,nabt)
      implicit none

      integer,intent(in) :: io_s, io_e, Nspin

      integer,intent(in) :: is_array(3),ie_array(3),is(3),ie(3)
      integer,intent(in) :: idx(is(1)-4:ie(1)+4),idy(is(2)-4:ie(2)+4),idz(is(3)-4:ie(3)+4)
      integer,intent(in) :: igs(3),ige(3)

      complex(8),intent(in)  :: tpsi   (is_array(1):ie_array(1),is_array(2):ie_array(2),is_array(3):ie_array(3),1:Nspin,io_s:io_e)
      complex(8),intent(out) :: htpsi  (is_array(1):ie_array(1),is_array(2):ie_array(2),is_array(3):ie_array(3),1:Nspin,io_s:io_e)
      ! type(s_scalar) ,intent(in) :: V_local(is(1):ie(1),is(2):ie(2),is(3):ie(3), 1:Nspin)
      type(s_scalar) ,intent(in) :: V_local(1:Nspin)
      real(8),   intent(in)  :: lap0
      real(8),   intent(in)  :: lapt(12), nabt(12)

      complex(8), parameter :: zI=(0.d0,1.d0)

      integer    :: ix,iy,iz,io,ispin
      complex(8) :: v,w
      !complex(8) :: t(8)
      complex(8) :: t_0,t_1

#ifdef __INTEL_COMPILER
#if defined(__KNC__) || defined(__AVX512F__)
#   define MEM_ALIGN   64
#   define VECTOR_SIZE 4
# else
#   define MEM_ALIGN   32
#   define VECTOR_SIZE 2
# endif

    ! !dir$ assume_aligned V_local:MEM_ALIGN
    ! !dir$ assume_aligned tpsi   :MEM_ALIGN
    ! !dir$ assume_aligned htpsi  :MEM_ALIGN
#endif

#define DX(dt) idx(ix+(dt)),iy,iz
#define DY(dt) ix,idy(iy+(dt)),iz
#define DZ(dt) ix,iy,idz(iz+(dt))
    !$acc parallel copyin(V_local, tpsi, htpsi)
    !$acc loop collapse(5)
      do io=io_s,io_e
      do ispin=1,Nspin

      do iz=igs(3),ige(3)
      do iy=igs(2),ige(2)

    ! !dir$ assume_aligned V_local(is(1),iy,iz):MEM_ALIGN
    ! !dir$ assume_aligned tpsi(is_array(1),iy,iz)   :MEM_ALIGN
    ! !dir$ assume_aligned htpsi(is_array(1),iy,iz)  :MEM_ALIGN

      do ix=igs(1),ige(1)
        t_0 = tpsi(DX( 1), ispin, io)
        t_1 = tpsi(DX(-1), ispin, io)
        v=lapt(1)*(t_0+t_1)
        w=nabt(1)*(t_0-t_1)

        t_0 = tpsi(DX( 2), ispin, io)
        t_1 = tpsi(DX(-2), ispin, io)
        v=lapt(2)*(t_0+t_1) + v
        w=nabt(2)*(t_0-t_1) + w

        t_0 = tpsi(DX( 3), ispin, io)
        t_1 = tpsi(DX(-3), ispin, io)
        v=lapt(3)*(t_0+t_1) + v
        w=nabt(3)*(t_0-t_1) + w

        t_0 = tpsi(DX( 4), ispin, io)
        t_1 = tpsi(DX(-4), ispin, io)
        v=lapt(4)*(t_0+t_1) + v
        w=nabt(4)*(t_0-t_1) + w

        t_0 = tpsi(DY( 1), ispin, io)
        t_1 = tpsi(DY(-1), ispin, io)
        v=lapt(5)*(t_0+t_1) + v
        w=nabt(5)*(t_0-t_1) + w

        t_0 = tpsi(DY( 2), ispin, io)
        t_1 = tpsi(DY(-2), ispin, io)
        v=lapt(6)*(t_0+t_1) + v
        w=nabt(6)*(t_0-t_1) + w
        
        t_0 = tpsi(DY( 3), ispin, io)
        t_1 = tpsi(DY(-3), ispin, io)
        v=lapt(7)*(t_0+t_1) + v
        w=nabt(7)*(t_0-t_1) + w

        t_0 = tpsi(DY( 4), ispin, io)
        t_1 = tpsi(DY(-4), ispin, io)
        v=lapt(8)*(t_0+t_1) + v
        w=nabt(8)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 1), ispin, io)
        t_1 = tpsi(DZ(-1), ispin, io)
        v=lapt( 9)*(t_0+t_1) + v
        w=nabt( 9)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 2), ispin, io)
        t_1 = tpsi(DZ(-2), ispin, io)
        v=lapt(10)*(t_0+t_1) + v
        w=nabt(10)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 3), ispin, io)
        t_1 = tpsi(DZ(-3), ispin, io)
        v=lapt(11)*(t_0+t_1) + v
        w=nabt(11)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 4), ispin, io)
        t_1 = tpsi(DZ(-4), ispin, io)
        v=lapt(12)*(t_0+t_1) + v
        w=nabt(12)*(t_0-t_1) + w

        htpsi(ix,iy,iz,ispin,io) = V_local(ispin)%f(ix,iy,iz)*tpsi(ix,iy,iz,ispin,io) &
                        + lap0*tpsi(ix,iy,iz,ispin,io) &
                        - 0.5d0 * v - zI * w
      end do

      end do
      end do

      end do
      end do
    !$acc end parallel
    end subroutine

    subroutine zstencil_typical_gpu(io_s,io_e,Nspin,is_array,ie_array,is,ie,idx,idy,idz,igs,ige &
                                   ,tpsi,htpsi,V_local,lap0,lapt,nabt)
      implicit none

      integer,intent(in) :: io_s, io_e, Nspin

      integer,intent(in) :: is_array(3),ie_array(3),is(3),ie(3)
      integer,intent(in) :: idx(is(1)-4:ie(1)+4),idy(is(2)-4:ie(2)+4),idz(is(3)-4:ie(3)+4)
      integer,intent(in) :: igs(3),ige(3)

      complex(8),intent(in)  :: tpsi   (is_array(1):ie_array(1),is_array(2):ie_array(2),is_array(3):ie_array(3),1:Nspin,io_s:io_e)
      complex(8),intent(out) :: htpsi  (is_array(1):ie_array(1),is_array(2):ie_array(2),is_array(3):ie_array(3),1:Nspin,io_s:io_e)
      ! type(s_scalar) ,intent(in) :: V_local(is(1):ie(1),is(2):ie(2),is(3):ie(3), 1:Nspin)
      type(s_scalar) ,intent(in) :: V_local(1:Nspin)
      real(8),   intent(in)  :: lap0
      real(8),   intent(in)  :: lapt(12), nabt(12)

      complex(8), parameter :: zI=(0.d0,1.d0)

      integer    :: ix,iy,iz,io,ispin
      complex(8) :: v,w
      !complex(8) :: t(8)
      complex(8) :: t_0,t_1

#ifdef __INTEL_COMPILER
#if defined(__KNC__) || defined(__AVX512F__)
#   define MEM_ALIGN   64
#   define VECTOR_SIZE 4
# else
#   define MEM_ALIGN   32
#   define VECTOR_SIZE 2
# endif

    ! !dir$ assume_aligned V_local:MEM_ALIGN
    ! !dir$ assume_aligned tpsi   :MEM_ALIGN
    ! !dir$ assume_aligned htpsi  :MEM_ALIGN
#endif

#define DX(dt) idx(ix+(dt)),iy,iz
#define DY(dt) ix,idy(iy+(dt)),iz
#define DZ(dt) ix,iy,idz(iz+(dt))
    !$acc parallel copyin(V_local, tpsi, htpsi)
    !$acc loop collapse(5)
!$omp target teams distribute map(to:V_local, tpsi, htpsi) map(from:htpsi)
      do io=io_s,io_e
      do ispin=1,Nspin

      do iz=igs(3),ige(3)
      do iy=igs(2),ige(2)

    ! !dir$ assume_aligned V_local(is(1),iy,iz):MEM_ALIGN
    ! !dir$ assume_aligned tpsi(is_array(1),iy,iz)   :MEM_ALIGN
    ! !dir$ assume_aligned htpsi(is_array(1),iy,iz)  :MEM_ALIGN

      do ix=igs(1),ige(1)
        t_0 = tpsi(DX( 1), ispin, io)
        t_1 = tpsi(DX(-1), ispin, io)
        v=lapt(1)*(t_0+t_1)
        w=nabt(1)*(t_0-t_1)

        t_0 = tpsi(DX( 2), ispin, io)
        t_1 = tpsi(DX(-2), ispin, io)
        v=lapt(2)*(t_0+t_1) + v
        w=nabt(2)*(t_0-t_1) + w

        t_0 = tpsi(DX( 3), ispin, io)
        t_1 = tpsi(DX(-3), ispin, io)
        v=lapt(3)*(t_0+t_1) + v
        w=nabt(3)*(t_0-t_1) + w

        t_0 = tpsi(DX( 4), ispin, io)
        t_1 = tpsi(DX(-4), ispin, io)
        v=lapt(4)*(t_0+t_1) + v
        w=nabt(4)*(t_0-t_1) + w

        t_0 = tpsi(DY( 1), ispin, io)
        t_1 = tpsi(DY(-1), ispin, io)
        v=lapt(5)*(t_0+t_1) + v
        w=nabt(5)*(t_0-t_1) + w

        t_0 = tpsi(DY( 2), ispin, io)
        t_1 = tpsi(DY(-2), ispin, io)
        v=lapt(6)*(t_0+t_1) + v
        w=nabt(6)*(t_0-t_1) + w
        
        t_0 = tpsi(DY( 3), ispin, io)
        t_1 = tpsi(DY(-3), ispin, io)
        v=lapt(7)*(t_0+t_1) + v
        w=nabt(7)*(t_0-t_1) + w

        t_0 = tpsi(DY( 4), ispin, io)
        t_1 = tpsi(DY(-4), ispin, io)
        v=lapt(8)*(t_0+t_1) + v
        w=nabt(8)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 1), ispin, io)
        t_1 = tpsi(DZ(-1), ispin, io)
        v=lapt( 9)*(t_0+t_1) + v
        w=nabt( 9)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 2), ispin, io)
        t_1 = tpsi(DZ(-2), ispin, io)
        v=lapt(10)*(t_0+t_1) + v
        w=nabt(10)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 3), ispin, io)
        t_1 = tpsi(DZ(-3), ispin, io)
        v=lapt(11)*(t_0+t_1) + v
        w=nabt(11)*(t_0-t_1) + w

        t_0 = tpsi(DZ( 4), ispin, io)
        t_1 = tpsi(DZ(-4), ispin, io)
        v=lapt(12)*(t_0+t_1) + v
        w=nabt(12)*(t_0-t_1) + w

        htpsi(ix,iy,iz,ispin,io) = V_local(ispin)%f(ix,iy,iz)*tpsi(ix,iy,iz,ispin,io) &
                        + lap0*tpsi(ix,iy,iz,ispin,io) &
                        - 0.5d0 * v - zI * w
      end do

      end do
      end do

      end do
      end do
    !$acc end parallel
    end subroutine

end module func

program main
    use func
    integer :: is_ary(3), ie_ary(3), is(3), ie(3), igs(3), ige(3)
    integer :: idx(-3:24), idy(-3:40), idz(-3:56)
    !integer, allocatable:: idx(:),idy(:),idz(:)
    complex(8), allocatable :: htpsi(:,:,:,:,:), tpsi(:,:,:,:,:), htpsi_cpu(:,:,:,:,:)
    integer :: Nspin = 1
    integer :: io_s, io_e, cnt = 0
    type(s_scalar) :: V_local(1)
    real(8) :: lap0 = 50.68313352342233
    real(8) :: lapt(12), nabt(12)

    io_s = 1
    io_e = 52
    is_ary(1)=1
    is_ary(2)=1
    is_ary(3)=1
    ie_ary(1)=20
    ie_ary(2)=37
    ie_ary(3)=52
    is(1)=1
    is(2)=1
    is(3)=1
    ie(1)=20
    ie(2)=36
    ie(3)=52
    igs(1)=1
    igs(2)=1
    igs(3)=1
    ige(1)=20
    ige(2)=36
    ige(3)=52
    allocate(htpsi(is_ary(1):ie_ary(1),is_ary(2):ie_ary(2),is_ary(3):ie_ary(3),1:Nspin,io_s:io_e))
    allocate(htpsi_cpu(is_ary(1):ie_ary(1),is_ary(2):ie_ary(2),is_ary(3):ie_ary(3),1:Nspin,io_s:io_e))
    allocate(tpsi(is_ary(1):ie_ary(1),is_ary(2):ie_ary(2),is_ary(3):ie_ary(3),1:Nspin,io_s:io_e))
    allocate(V_local(1)%f(is_ary(1):ie_ary(1),is_ary(2):ie_ary(2),is_ary(3):ie_ary(3)))
    !allocate(idx(-3,24))
    !allocate(idy(-3,40))
    !allocate(idz(-3,56))
    do m=io_s,io_e
        do l=1,Nspin
            do k=is_ary(3),ie_ary(3)
                do j=is_ary(2),ie_ary(2)
                    do i=is_ary(1),ie_ary(1)
                        tpsi(i,j,k,l,m) = 0.5d0
                    enddo
                enddo
            enddo
        enddo
    enddo

    do i=1,12
        lapt(i)=1.0d0
        nabt(i)=1.0d0
    enddo
    do i = -3,ie(1)+4
        idx(i)=i
        if (i .le. 0) idx(i) = idx(i) + ie(1)
        if (i .gt. ie(1)) idx(i) = idx(i) - ie(1)
    enddo
    do i = -3,ie(2)+4
        idy(i)=i
        if (i .le. 0) idy(i) = idy(i) + ie(2)
        if (i .gt. ie(2)) idy(i) = idy(i) - ie(2)
    enddo
    do i = -3,ie(3)+4
        idz(i)=i
        if (i .le. 0) idz(i) = idz(i) + ie(3)
        if (i .gt. ie(3)) idz(i) = idz(i) - ie(3)
    enddo
    call zstencil_typical(io_s,io_e,1,is_ary,ie_ary,is,ie,idx,idy,idz,igs,ige, &
                       tpsi,htpsi,V_local,lap0,lapt,nabt)
#ifdef USE_GPU
    call zstencil_typical_gpu(io_s,io_e,1,is_ary,ie_ary,is,ie,idx,idy,idz,igs,ige, &
                       tpsi,htpsi_cpu,V_local,lap0,lapt,nabt)
#else
    call zstencil_typical(io_s,io_e,1,is_ary,ie_ary,is,ie,idx,idy,idz,igs,ige, &
                       tpsi,htpsi_cpu,V_local,lap0,lapt,nabt)
#endif
    do m=io_s,io_e
        do l=1,Nspin
            do k=is_ary(3),ie_ary(3)
                do j=is_ary(2),ie_ary(2)
                    do i=is_ary(1),ie_ary(1)
                        !if ((.not. htpsi(i,j,k,l,m) .eq. htpsi_cpu(i,j,k,l,m)) .or. cnt .lt. 10) then
                        if (cnt .lt. 10) then
                            print *, i,j,k,l,m,htpsi(i,j,k,l,m),htpsi_cpu(i,j,k,l,m)
                        endif
                        cnt = cnt + 1
                    enddo
                enddo
            enddo
        enddo
    enddo
end program
