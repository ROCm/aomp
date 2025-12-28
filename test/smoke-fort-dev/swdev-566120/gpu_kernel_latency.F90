program gpu_kernel_latency
  use gpu_kernel_latency_mod
#ifdef _OPENMP
  use omp_lib
#endif
  implicit none
  integer, parameter :: nproma=10000, nlev=40, nb=100, niter=1000
  ! integer, parameter :: nproma=5, nlev=4, nb=1, niter=1
  integer :: i, j, k, map_id, ind_s, ind_e, it, nteams

  real(kind=real64), allocatable :: arr(:,:,:), arr_out_ref(:,:,:)
  real(kind=real64), allocatable :: arr_out(:,:,:)
  real(kind=real64) :: t_tot, t_avg_launch
  real(kind=real64) :: t_start, t_end
#ifndef USE_NOWAIT
  character(len=*), parameter :: nowait_str = ''
#else
  character(len=*), parameter :: nowait_str = 'nowait '
#endif
  
  allocate(arr(nproma, nlev, nb), &
      arr_out_ref(nproma, nlev, nb))
  arr_out_ref=0
  
  allocate(arr_out(nproma, nlev, nb))
  arr_out=0
    
  ! Initialize data
  do k=1,nb
    do j=1,nlev
      do i=1,nproma
        arr(i,j,k)=((-1)**(mod(i,2)+1))*real(nproma*nb*(k-1)+nproma*(j-1)+i, real64) ! Use linear index as data
        arr_out(i,j,k)=0
        arr_out_ref(i,j,k)=0
      end do
    end do
  end do
  
  ind_s=1
  ind_e=nproma
  
  ! Compute reference result
  call gpu_kernel_latency_ref(nproma, nlev, nb, ind_s, ind_e, arr, arr_out_ref)
  
!$omp target data map(to: arr) map(tofrom: arr_out)

  ! Warmup
#ifndef USE_NOWAIT
  call gpu_kernel_latency_basic(nproma, nlev, nb, ind_s, ind_e, arr, arr_out, .true.)
#else
  call gpu_kernel_latency_nowait(nproma, nlev, nb, ind_s, ind_e, arr, arr_out, .true.)
#endif
  ! Benchmark
  t_start=ftimer()
  do it=1,niter
#ifndef USE_NOWAIT
  call gpu_kernel_latency_basic(nproma, nlev, nb, ind_s, ind_e, arr, arr_out, .true.)
#else
  call gpu_kernel_latency_nowait(nproma, nlev, nb, ind_s, ind_e, arr, arr_out, .true.)
#endif
  end do
  t_end=ftimer()
  t_tot=t_end-t_start
  t_avg_launch=t_tot/(niter*nb)

!$omp end target data
  
#if 0
  do k=1,nb
     write (*,*) 'k=', k
     write (*,*) 'arr_out:'
     call print_2d_array(arr_out(:,:,k))
     write (*,*) 'arr_out_ref:'
     call print_2d_array(arr_out_ref(:,:,k))
#ifdef USE_NOWAIT
     write (*,*) 'arr_out_nowait:'
     call print_2d_array(arr_out_nowait(:,:,k))
#endif
  end do
#endif
  
  ! Output timing
  write (*,'(A)') 'Parameters:'
  write (*,'(X,A,I0)') 'nproma=', nproma
  write (*,'(X,A,I0)') 'nlev=', nlev
  write (*,'(X,A,I0)') 'nb=', nb
  write (*,'(X,A,I0)') 'niter=', niter
  write (*,'(A,A,A,F18.5)') 'Time ', nowait_str, '(s)=', t_tot
  write (*,'(A,A,A,F18.5)') 'Time ', nowait_str, 'per kernel (us)=', t_avg_launch*1e6
  
  ! Validate results
  if (all(abs(arr_out_ref-arr_out)/abs(arr_out_ref) < epstol)) then
    write (*,'(A)') 'arr_out validates'
  else
    write (*,'(A)') 'ERROR: arr_out does not validate'
  end if
  deallocate(arr, arr_out_ref)

  deallocate(arr_out)

contains

  function ftimer() result(timerval)
    use iso_fortran_env, only : real64, int64
    implicit none
    real(kind=real64) :: timerval
    
#ifndef _OPENMP
    integer(kind=int64) :: t, rate
    call system_clock(t,count_rate=rate)
    timerval = real(t,real64)/real(rate,real64)
#else
    timerval = omp_get_wtime()
#endif    
  end function ftimer

  subroutine print_2d_array(array)
    real(kind=real64), intent(in) :: array(:,:)

    integer :: i, j
    
    do i=1,size(array,1)
       do j=1,size(array,2)
          write (*,'(F18.5,X)', ADVANCE='NO') array(i,j)
       end do
       write (*,'(A)')
    end do
  end subroutine print_2d_array
  
end program gpu_kernel_latency
