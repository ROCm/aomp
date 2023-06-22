module LaunchLatencyUtils
    use, intrinsic :: ISO_Fortran_env, only: REAL64,INT64

    implicit none

    integer:: num_times  = 100000
    integer:: selection  = 1
    contains

        function get_wtime() result(t)
#if defined(USE_OMP_GET_WTIME)
          use omp_lib
          implicit none
          real(kind=REAL64) :: t
          t = omp_get_wtime()
#elif  defined(USE_CPU_TIME)
          implicit none
          real(kind=REAL64) :: t
          real :: r
          call cpu_time(r)
          t = r
#else
          implicit none
          real(kind=REAL64) :: t
          integer(kind=INT64) :: c, r
          call system_clock(count = c, count_rate = r)
          t = real(c,REAL64) / real(r,REAL64)
#endif
        end function get_wtime


        subroutine Empty()
            implicit none
            integer(kind=4) :: A
            !$omp target 
               A=0
            !$omp end target
        end subroutine Empty


subroutine run_all(timings)
            implicit none
            real(kind=REAL64), intent(inout) :: timings
            real(kind=REAL64) :: t1, t2
            integer(kind=INT64) :: i

            t1 = get_wtime()
            do i=1,num_times
                call Empty()

            end do
            t2 = get_wtime()
            timings = t2-t1
        end subroutine run_all
end module LaunchLatencyUtils

program LaunchLatency
    use LaunchLatencyUtils
    use, intrinsic :: ISO_Fortran_env, only: REAL64,INT64
    real(kind=REAL64) :: timings       
    integer :: argc, i
    character(len=64) :: argtmp
    integer :: arglen,err

    argc = command_argument_count()
    if (argc .gt. 0) then
      call get_command_argument(1,argtmp,arglen,err)
      read(argtmp,'(i15)') num_times
    endif
    write(*,*)  "Launch Latency Fortran", num_times

    timings = -1.0d0
    call run_all(timings)
    write(*,'(a, f14.9)') "Latncy per launch:", timings/ num_times

end program LaunchLatency
