module gpu_kernel_latency_mod
  use iso_fortran_env, only : real64
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none

  real(kind=real64) :: epstol = 100*tiny(real(1,real64))

#ifdef USE_ROCTX
  interface
    ! int roctxRangePushA(const char* message);
    integer(kind=c_int) function roctxRangePushA(message) &
        bind(C, name='roctxRangePushA')
      use iso_c_binding, only: c_int, c_char
      implicit none
      character(kind=c_char) :: message(*)
    end function roctxRangePushA
    
    !  int roctxRangePop();
    integer(kind=c_int) function roctxRangePop() &
        bind(C, name='roctxRangePop')
      use iso_c_binding, only: c_int
      implicit none
    end function roctxRangePop
  end interface
#endif
#ifdef USE_NVTX
  interface
    ! int nvtxRangePushA(const char* message);
    integer(kind=c_int) function nvtxRangePushA(message) &
        bind(C, name='nvtxRangePushA')
      use iso_c_binding, only: c_int, c_char
      implicit none
      character(kind=c_char) :: message(*)
    end function nvtxRangePushA
    
    !  int nvtxRangePop();
    integer(kind=c_int) function nvtxRangePop() &
        bind(C, name='nvtxRangePop')
      use iso_c_binding, only: c_int
      implicit none
    end function nvtxRangePop
  end interface
#endif
contains

#ifndef USE_NOWAIT
  subroutine gpu_kernel_latency_basic(nproma, nlev, nb, idx_s, idx_e, arr_in, arr_out, lzacc)
    integer, intent(in) :: nproma, nlev, nb, idx_s, idx_e
    real(kind=real64), intent(in) :: arr_in(nproma,nlev,nb)
    real(kind=real64), intent(out) :: arr_out(nproma,nlev,nb)
    logical :: lzacc

    integer :: jec, jb, jk, ji, rocstat, nvstat
    character(len=*), parameter :: kernel_name = 'gpu_kernel_latency_basic'
    
#ifdef USE_ROCTX
    rocstat = roctxRangePushA(kernel_name)
#endif
#ifdef USE_NVTX
    nvstat = nvtxRangePushA(kernel_name)
#endif
    do jb=1,nb
!$omp target teams loop collapse(2) &
!$omp if(lzacc)&
!$omp ! nowait
      do jk = 1,nlev
        do ji=idx_s,idx_e
          arr_out(ji,jk,jb)=arr_in(ji,jk,jb) 
        end do
      end do
    end do
#ifdef USE_ROCTX
    rocstat = roctxRangePop()
#endif
#ifdef USE_NVTX
    nvstat = nvtxRangePop()
#endif
    
!$omp taskwait
  end subroutine gpu_kernel_latency_basic
#else
  subroutine gpu_kernel_latency_nowait(nproma, nlev, nb, idx_s, idx_e, arr_in, arr_out, lzacc)
    integer, intent(in) :: nproma, nlev, nb, idx_s, idx_e
    real(kind=real64), intent(in) :: arr_in(nproma,nlev,nb)
    real(kind=real64), intent(out) :: arr_out(nproma,nlev,nb)
    logical :: lzacc

    integer :: jec, jb, jk, ji, rocstat, nvstat
    character(len=*), parameter :: kernel_name = 'gpu_kernel_latency_nowait'
    
#ifdef USE_ROCTX
    rocstat = roctxRangePushA(kernel_name)
#endif
#ifdef USE_NVTX
    nvstat = nvtxRangePushA(kernel_name)
#endif
    do jb=1,nb
!$omp target teams loop collapse(2) &
!$omp if(lzacc)&
!$omp nowait
      do jk = 1,nlev
        do ji=idx_s,idx_e
          arr_out(ji,jk,jb)=arr_in(ji,jk,jb) 
        end do
      end do
    end do
#ifdef USE_ROCTX
    rocstat = roctxRangePop()
#endif
#ifdef USE_NVTX
    nvstat = nvtxRangePop()
#endif
    
!$omp taskwait
  end subroutine gpu_kernel_latency_nowait
#endif // USE_NOWAIT
  
  subroutine gpu_kernel_latency_ref(nproma, nlev, nb, idx_s, idx_e, arr_in, arr_out)
    integer, intent(in) :: nproma, nlev, nb, idx_s, idx_e
    real(kind=real64), intent(in) :: arr_in(nproma, nlev, nb)
    real(kind=real64), intent(out) :: arr_out(nproma, nlev, nb)

    integer :: jec, jb, jk, ji

    ! From ICON
    do jb=1,nb
       do jk = 1, nlev
          do ji=idx_s,idx_e
             arr_out(ji,jk,jb)=arr_in(ji,jk,jb)
          end do
       end do
    end do
  end subroutine gpu_kernel_latency_ref

end module gpu_kernel_latency_mod
