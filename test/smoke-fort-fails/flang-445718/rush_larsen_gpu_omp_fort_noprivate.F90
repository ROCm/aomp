! Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
! Goulash project contributors LLNL-CODE-795383, All rights reserved.
! For details about use and distribution, please read LICENSE and NOTICE from
! the Goulash project repository: http://github.com/llnl/goulash
! SPDX-License-Identifier: BSD-3-Clause
!
! Designed to allow direct performance comparisons between
! naively written HIP/CUDA and OpenMP GPU offloading schemes
! in a variety of coding styles and languages of a
! parameterized embarrassingly parallel Rush Larsen kernel.
! Also allows testing build systems (including SPACK) handling
! of complicated build situations that LLNL cares about.
!
! Designed to create several single file test variants
! where no -D options required to select the variant
! and no include files are needed.
! As of goulash 2.1, the finalize_source script
! is used to generate customized source files
! for each test, from a common source. 
!
! The key computational kernel can be located by
! searching for: KERNEL
!
! Designed to create trivial variants of the same
! test to be compiled with different compilers for
! the interoperability tests.   This is why most
! functions are static and the main kernel test
! is called rush_larsen_gpu_omp_fort (preprocessor
! defines this and VARIANT_DESC based on
! variant configuration).
!
! The naming convention of the variant copies is
! intended to indicate variant abilities:
! _cpu_serial      - single threaded, no OpenMP, on CPU
! _cpu_omp         - use OpenMP to spawn threads on CPU
! _gpu_omp         - uses OpenMP to offload to GPU
! _gpu_hip         - uses HIP to offload to AMD or Nvidia GPU
! _gpu_lambda_hip  - RAJA-like lambda HIP variant
! _gpu_cuda        - uses CUDA to offload to Nvidia GPU
! _gpu_lambda_cuda - RAJA-like lambda CUDA variant
! *_mpi            - uses and exercises MPI e.g. _gpu_omp_mpi
! *_fort           - Fortran version e.g. _gpu_omp_mpi_fort
!
! For the interop tests, there is an additional suffix
! to indicate different copies of the same configuration
! that are intended to be compiled by different compilers:
! _compiler1  - E.g., rush_larsen_gpu_omp_compiler1.cc
!
! VARIANT_DESC set by preprocessor directives to
! the configuration of this file.
!
! Recommended that a -DCOMPILERID be set to the compiler used to compile each file:
!
! /opt/rocm-4.0.1/llvm/bin/clang++  -o rush_larsen_cpu_omp -O3 -g "-DCOMPILERID=rocm-4.0.1" -fopenmp rush_larsen_cpu_omp.cc
!
! Run with no arguments for suggested arguments, for example:
!   Usage: ./rush_larsen_cpu_omp  Iterations  Kernel_GBs_used
!
!     Measure serial launch overhead:  env OMP_NUM_THREADS=1 ./rush_larsen_cpu_omp 100000 .00000001
!     Measure launch overhead:         ./rush_larsen_cpu_omp 100000 .00000001
!     Measure kernel performance:      ./rush_larsen_cpu_omp    100 10
!
! The Goulash project conceived of and designed by David Richards,
! Tom Scogland, and John Gyllenhaal at LLNL Oct 2019.
! Please contact John Gyllenhaal (gyllenhaal1@llnl.gov) with questions.
!
! Rush Larsen core CUDA/OpenMP kernels written by Rob Blake (LLNL) Sept 2016.
! The goulash Rush Larsen tests add benchmarking infrastructure
! around this incredibly useful compact GPU test kernel.   Thank you Rob!
!
! Inline performance measurements added (nvprof not needed)
! by John Gyllenhaal at LLNL 11/10/20.
!
! Command line argument handling, performance difference printing in
! form easily found with grep, OpenMP thread mapping and initial data
! sanity checks on just the first array element calculated by kernel
! by John Gyllenhaal at LLNL 03/22/21
!
! Pulled code from print_openmp_mapping.c by John Gyllenhaal at
! LLNL written June 2020 which was based on mpibind tests
! (https://github.com/LLNL/mpibind) by Edgar Leon at LLNL
!
! RAJA-perf-suite-like (https://github.com/LLNL/RAJAPerf)
! C++ lambda versions created by Jason Burmark at LLNL 06/16/21
!
! C-like Fortran ports (by hand) of C++ version of
! rush larsen variants by John Gyllenhaal at LLNL 06/28/21.
!
! MPI stat aggregation and MPI exercisers written
! by John Gyllenhaal at LLNL based on previous user issue reproducers.
! Pulled into rush larsen tests on 07/02/21 by John Gyllenhaal at LLNL.
!
! Enhanced data checks of all kernel generated array data, including
! across MPI ranks by John Gyllenhaal at LLNL 07/03/21
!
! Interop versions create by John Gyllenhaal at LLNL 07/14/21
! to test mixing all the Rush Larsen tests with multiple GPU compilers
! all in one final executable.
!
! Initial test generator from template files, including Makefiles
! created by John Gyllenhaal at LLNL 07/21/21 for V2.0RC1
!
! V2.0 RC1 07/21/21 Added MPI support, interop version, enhanced data checks.
! V1.2 06/28/21 Fortran and C++ lambda versions added, consistent use of long type
! V1.1 03/22/21 command line args, perf diffs, maps, checks return codes and answer
! V1.0 11/10/20 initial release, hard coded inputs, no error checking

! Allow version to be printed in output 
#define VERSION_STRING "Version 2.0 RC1 (7/21/21)"

! Sets up and runs the doRushLarsen kernel 'iterations' times, 
! allocating CPU arrays and perhaps GPU arrays to consume 
! kernel_mem_used GBs of memory.
!
! This polynomial is a fit to the dynamics of a small part of a cardiac
! myocyte, specifically the fast sodium m-gate described here:
! https://www.ncbi.nlm.nih.gov/pubmed/16565318
!
! Does exactly the same work on every cell.   Can scale from one cell
! to filling entire memory.   Does use cell's value as input
! to calculations.
!
! Returns number of data check failures, returns 0 if all data checks out.
function rush_larsen_gpu_omp_fort(iterations_, kernel_mem_used_)

  ! Only include OpenMP for variants that need it
  use omp_lib

  ! Get mappings to stdout, etc. so can flush output
  use, intrinsic :: iso_fortran_env, only : stdin=>input_unit, &
       &                                    stdout=>output_unit, &
       &                                    stderr=>error_unit

  ! Catch misspelled variables 
  implicit none

  ! Declare arguments
  integer :: rush_larsen_gpu_omp_fort
  integer(8),  intent(IN) :: iterations_
  integer(8) :: iterations
  real(8), intent(IN) :: kernel_mem_used_
  real(8) :: kernel_mem_used

  ! Declare local variables
  ! NOTE: All subroutines and functions called by this routine
  !       can access these variables!
  !       Used for variant_desc, kernel_mem_used_str, timestamp, sec_str, us_str, ierr
  character(1024) :: variant_desc
  character(50) :: timestamp
  character(50) :: kernel_mem_used_str
  character(50) :: sec_str, us_str, transfer_str
  real(8) :: kernel_starttime, kernel_endtime, kernel_runtime, base_time, last_timestamp, cur_secs
  real(8) :: transfer_starttime, transfer_endtime, transfer_runtime
  integer(8) :: nCells, status_point
  integer :: rank = 0 ! Rank will be 0 for the no MPI case 
  integer :: ierr
  integer(8) :: fail_count = 0
  real(8) :: sum1, sum2, x, mhu, tauR
  integer(8) :: itime, ii
  integer(4) :: j, k
  real(8), allocatable :: m_gate(:), Vm(:)
  integer(4), parameter :: Mhu_l = 10
  integer(4), parameter :: Mhu_m = 5
  integer(4), parameter :: Tau_m = 18
  ! Must use 'd' in every constant in order to get full real*8 values and matching results
  real(8) :: Mhu_a(0:14) = (/&
       &  9.9632117206253790d-01,  4.0825738726469545d-02,  6.3401613233199589d-04,&
       &  4.4158436861700431d-06,  1.1622058324043520d-08,  1.0000000000000000d+00,&
       &  4.0568375699663400d-02,  6.4216825832642788d-04,  4.2661664422410096d-06,&
       &  1.3559930396321903d-08, -1.3573468728873069d-11, -4.2594802366702580d-13,&
       &  7.6779952208246166d-15,  1.4260675804433780d-16, -2.6656212072499249d-18/)
  ! Must use 'd' in every constant in order to get full real*8 values and matching results
  real(8) :: Tau_a(0:18) = (/&
       &  1.7765862602413648d+01*0.02d+00,  5.0010202770602419d-02*0.02d+00, -7.8002064070783474d-04*0.02d+00,&
       & -6.9399661775931530d-05*0.02d+00,  1.6936588308244311d-06*0.02d+00,  5.4629017090963798d-07*0.02d+00,&
       & -1.3805420990037933d-08*0.02d+00, -8.0678945216155694d-10*0.02d+00,  1.6209833004622630d-11*0.02d+00,&
       &  6.5130101230170358d-13*0.02d+00, -6.9931705949674988d-15*0.02d+00, -3.1161210504114690d-16*0.02d+00,&
       &  5.0166191902609083d-19*0.02d+00,  7.8608831661430381d-20*0.02d+00,  4.3936315597226053d-22*0.02d+00,&
       & -7.0535966258003289d-24*0.02d+00, -9.0473475495087118d-26*0.02d+00, -2.9878427692323621d-28*0.02d+00,&
       &  1.0000000000000000d+00/)

  ! Allow compiler to be passed in at compile time
  ! Must pass in quoted string on command line, i.e., '-DCOMPILERID="CCE"' 
#if defined(COMPILERID)
  write (variant_desc, 20) "gpu_omp_fort", ' [', COMPILERID , ']'
20 format(a, a,a,a)
#else
  variant_desc="gpu_omp_fort"
#endif

  ! To make interop performance easier to compare,
  ! start this file's timers over every time called.
  !
  ! Reset this file's secs_elapsed() counter to 0 
  cur_secs = get_raw_secs()
  base_time = get_base_time(cur_secs)

  ! Synchronize printf timestamps across MPI ranks
  last_timestamp = get_last_timestamp(secs_elapsed())

  if (rank == 0) then
     ! Print separator before and after output with function name
     call get_timestamp_string(timestamp)
     print '(a,"--------------- Begin rush_larsen_",a," (timer zeroed) ---------------")', &
          & trim(timestamp), trim(variant_desc)
     flush(stdout)
  end if

  ! For print niceness, make .00000001 lower bound on GB memory
  if (kernel_mem_used_ < .00000001) then
     kernel_mem_used = .00000001
  else
     kernel_mem_used = kernel_mem_used_
  end if

  ! Calculate nCells from target memory target
  nCells =  ((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) / (8 * 2))

  ! Must have at least 1 cell 
  if (nCells < 1) then
     nCells = 1
  end if

  ! Must have at least 1 iteration 
  if (iterations_ < 1) then
     iterations=1
  else
     iterations=iterations_
  end if

  ! Give status every 10% of iterations 
  status_point=iterations/10
  ! Must be at least 1 to make mod work
  if (status_point < 1) then
     status_point = 1
  end if

  ! Print what we are running
  ! Convert kernel_mem_used to left justified string with leading 0
  ! This str is used in other subroutines and functions
  write (kernel_mem_used_str, 50) kernel_mem_used
50 format(F16.8)
  kernel_mem_used_str=adjustl(kernel_mem_used_str)
  ! This kernel_mem_used_str used in several other messages as is

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," START Rush Larsen ",i0," iters ",i0," cells ",a," GBs ",a)',&
          & trim(timestamp),iterations, nCells, trim(kernel_mem_used_str), trim(variant_desc)

     call get_timestamp_string(timestamp)
     print '(a," ",a)', trim(timestamp), trim(VERSION_STRING)
     flush(stdout)
  end if

  ! If using OpenMP offloading, make sure GPU works before doing test
  call verify_gpu_openmp(0)

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," ",a)', trim(timestamp), "Allocating and initializing kernel arrays"
     flush(stdout)
  end if

  ! Porting from C, so make all arrays start at index 0 to make port easier
  allocate(m_gate(0:nCells-1))
  m_gate=0.0

  ! Porting from C, so make all arrays start at index 0 to make port easier
  allocate(Vm(0:nCells-1))
  Vm=0.0

  ! No data transfer time if not using GPU 
  transfer_starttime=0.0
  transfer_endtime=0.0

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," Starting omp data map of CPU arrays to GPU")', trim(timestamp)
     flush(stdout)
  end if

  transfer_starttime=secs_elapsed()
  !$omp target enter data map(to: m_gate(0:nCells-1))
  !$omp target enter data map(to: Vm(0:nCells-1))
  !$omp target enter data map(to: Mhu_a(0:14))
  !$omp target enter data map(to: Tau_a(0:18))
  transfer_endtime=secs_elapsed()

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," Finished omp data map of CPU arrays to GPU")', trim(timestamp)
     flush(stdout)
  end if

  transfer_runtime=transfer_endtime-transfer_starttime

  ! Do the iterations asked for plus 1 for warmup
  do itime=0,iterations
     ! Print warmup message for 0th iteration
     if (itime == 0) then
        if (rank == 0) then
           call get_timestamp_string(timestamp)
           print '(a,a)', trim(timestamp), " Launching warmup iteration (not included in timings)"
           flush(stdout)
        end if
        ! Print status every 10% of iterations
     else if (modulo((itime-1), status_point) == 0) then
        if (itime == 1) then
           if (rank == 0) then
              call get_timestamp_string(timestamp)
              print '(a," Starting kernel timings for Rush Larsen ",i0," ",a)',&
                   & trim(timestamp),iterations, trim(kernel_mem_used_str)
              flush(stdout)
           end if
        end if

        if (rank == 0) then
           call get_timestamp_string(timestamp)
           print '(a,a,i6)', trim(timestamp), " Starting iteration ", itime
           flush(stdout)
        end if
     end if

     ! Start timer after warm-up iteration 0
     if (itime == 1) then
        kernel_starttime = secs_elapsed()
     end if

     !
     ! RUSH LARSEN KERNEL BEING TIMED START
     !
     ! Target GPU with OpenMP, data already mapped to GPU 
     !!!!$omp target teams distribute parallel do simd private(ii,x,sum1,j,sum2,k,mhu,tauR)
     !$omp target teams distribute parallel do
     do ii=0,nCells-1
        x = Vm(ii)
        sum1 = 0.0
        do j = Mhu_m-1, 0, -1
           sum1 = Mhu_a(j) + x*sum1
        end do
        sum2 = 0.0
        k = Mhu_m + Mhu_l - 1
        do j = k, Mhu_m, -1
           sum2 = Mhu_a(j) + x * sum2
        end do
        mhu = sum1/sum2

        sum1 = 0.0
        do j = Tau_m-1, 0, -1
           sum1 = Tau_a(j) + x*sum1
        end do
        tauR = sum1

        m_gate(ii) = m_gate(ii) + (mhu - m_gate(ii))*(1-exp(-tauR))
     end do
     ! End Target GPU with OpenMP, data already mapped to GPU 
     !$omp end target teams distribute parallel do 
     !!!$omp end target teams distribute parallel do simd 

     !
     ! RUSH LARSEN KERNEL BEING TIMED END
     ! 
  end do

  ! Get time after all iterations
  kernel_endtime = secs_elapsed ()

  ! Calculate kernel runtime
  kernel_runtime = kernel_endtime-kernel_starttime

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a,a,i0,a,a)',&
          & trim(timestamp)," Finished kernel timings for Rush Larsen ", iterations, " ", trim(kernel_mem_used_str) 
     flush(stdout)
  end if

  ! Print kernel runtime stats, syncs and aggregates MPI rank (if MPI mode)
  call print_runtime_stats(iterations, kernel_mem_used, kernel_runtime, transfer_runtime)

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a, " Starting omp target update of GPU result array to CPU array")', trim(timestamp)
     flush(stdout)
  end if

  ! Transfer GPU m_gate kernel memory to CPU kernel memory for data checks 
  !$omp target update from (m_gate(0:nCells-1))

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a, " Finished omp target update of GPU result array to CPU array")', trim(timestamp)
     flush(stdout)
  end if

  ! Do sanity and consistency checks on all of m_gate. Including cross-rank if in MPI mode.
  ! Prints PASS or FAIL based on data check results
  ! Returns fail count so can be returned to caller.
  fail_count = data_check (m_gate, iterations, kernel_mem_used, nCells)

  ! Free kernel GPU memory
  !$omp target exit data map(delete: m_gate(0:nCells-1))
  !$omp target exit data map(delete: Vm(0:nCells-1))
  !$omp target exit data map(delete: Mhu_a(0:14))
  !$omp target exit data map(delete: Tau_a(0:18))

  deallocate(Vm)
  deallocate(m_gate)

  if (rank == 0) then
     call get_timestamp_string(timestamp)
     print '(a," ",a,a)', trim(timestamp), "DONE Freed memory ", trim(variant_desc)
     flush(stdout)
     ! Print separator before and after output with function name
     call get_timestamp_string(timestamp)
     print '(a,"--------------- End rush_larsen_",a," ---------------")', &
          & trim(timestamp), trim(variant_desc)
     flush(stdout)
  end if

  ! Return number of data check failures
  rush_larsen_gpu_omp_fort = fail_count

contains

  ! Ends program either with MPI_Abort or STOP 1
  subroutine die()
    stop 1
  end subroutine die


  ! Get raw time in seconds as double (a large number).
  function get_raw_secs()
    ! Catch misspelled variables
    implicit none
    real(8) :: get_raw_secs
    integer(8) :: count, count_rate, count_max
    real(8) :: dcount, dcount_rate

    ! Get wall-clock time
    call system_clock(count, count_rate, count_max)
    dcount = count
    dcount_rate = count_rate
    ! Convert values to double (in seconds ) (a large number)
    get_raw_secs = dcount/dcount_rate
  end function get_raw_secs


  ! Returns base time.  If new_time > 0,
  ! sets base_time to new_time before returning.
  ! Using this as access method to static variable
  ! in a way I can trivially emulate in fortran.
  !
  ! Note: Lock shouldn't be needed, since even if multiple
  !       threads initialize this, it will be to basically
  !       the same value.
  !
  function get_base_time(new_time) 
    ! Catch misspelled variables
    implicit none
    real(8), intent(IN):: new_time
    real(8):: get_base_time
    real(8), save :: base_time = -1.0

    !If passed value > 0
    if (new_time > 0.0) then
       base_time = new_time
    end if

    get_base_time = base_time
  end function get_base_time


  ! Returns time in seconds (double) since the first call to secs_elapsed
  ! (i.e., the first call returns 0.0).
  function secs_elapsed ()
    ! Catch misspelled variables 
    implicit none
    real(8) :: secs_elapsed
    real(8) :: new_time, base_time

    ! Get current raw time (a big number)
    new_time = get_raw_secs()

    base_time = get_base_time(-1.0_8)

    ! If base time not set (negative), set to current time (pass in positive secs)
    if (base_time < 0.0) then
       base_time=get_base_time(new_time)
    end if

    ! Returned offset from first time called
    secs_elapsed = new_time - base_time
  end function secs_elapsed


  function get_last_timestamp(new_time) 
    ! Catch misspelled variables
    implicit none
    real(8), intent(IN):: new_time
    real(8):: get_last_timestamp
    real(8), save :: last_timestamp = -1.0

    !If passed value > 0
    if (new_time >= 0.0) then
       last_timestamp = new_time
    end if

    get_last_timestamp = last_timestamp
  end function get_last_timestamp


  ! Cannot wrap print in fortran so create utility function
  ! for creating timestamp prefix with diff from last timestamp.
  ! Generate timestamp string of this form:
  !    0.095 (0.000s): 
  subroutine get_timestamp_string (timestamp_string)
    ! Only include mpi for variants that need it
    ! Catch misspelled variables 
    implicit none
    character(len=*), intent(OUT) :: timestamp_string
    real(8) :: last_timestamp
    real(8) :: sec, diff
    integer :: rank = -1

    ! Get wall-clock time since first call to secs_elapsed
    sec = secs_elapsed ()

    ! Query last timestamp, set first time if needed
    last_timestamp = get_last_timestamp(-1.0_8)
    if (last_timestamp < 0.0) then
       last_timestamp=get_last_timestamp(sec)
    end if

    diff = sec - last_timestamp

    ! Set new last timestamp
    last_timestamp=get_last_timestamp(sec)

    ! No MPI case
    if (rank < 0) then
       ! Write out timestamp and diff seconds to buffer
       write (timestamp_string, 10) sec, ' (', diff, 's): '
10     format(f7.3,a,f5.3,a)

       ! MPI case, add rank
    else
       ! Write out timestamp and diff seconds to buffer
       write (timestamp_string, 11) rank, ": ", sec, ' (', diff, 's): '
11     format(i3,a,f7.3,a,f5.3,a)
    end if
  end subroutine get_timestamp_string


  ! If using OpenMP offloading, make sure GPU works before doing test 
  subroutine verify_gpu_openmp(gpu_id)
    use omp_lib
    integer, intent(in) :: gpu_id

    character(50) :: mpi_desc=""

    ! If using GPU, make sure GPU OpenMP gpu offloading works before doing test 
    integer:: runningOnGPU

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a," Selecting GPU ",i0, " as default device",a)', trim(timestamp), gpu_id, trim(mpi_desc)
       flush(stdout)
    end if

    ! Pick GPU to use to exercise selection call 
    call omp_set_default_device(gpu_id)

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a," Launching OpenMP GPU test kernel",a)', trim(timestamp), trim(mpi_desc)
       flush(stdout)
    end if

    ! Test if GPU is available using OpenMP4.5 legal code 
    runningOnGPU = 0
    !$omp target map(from:runningOnGPU)
    if (.not. omp_is_initial_device()) then
       runningOnGPU = 1
    else
       runningOnGPU = 2
    end if
    !$omp end target

    ! If still running on CPU, GPU must not be available, punt 
    if (runningOnGPU .ne. 1) then
       call get_timestamp_string(timestamp)
       print '(a," ", a, i0," ",a)', trim(timestamp), &
            & "ERROR: OpenMP GPU test kernel did NOT run on GPU ", gpu_id, trim(variant_desc)
       flush(stdout)
       call die()
    end if

    if (rank == 0) then
       call get_timestamp_string(timestamp)
       print '(a," Verified OpenMP target test kernel ran on GPU",a)', trim(timestamp), trim(mpi_desc)
       flush(stdout)
    end if
  end subroutine verify_gpu_openmp


  !Print kernel runtime stats and aggregate across MPI processes if necessary.
  !Prints one liner if not using MPI
  subroutine print_runtime_stats(iterations, kernel_mem_used, kernel_runtime, transfer_runtime)
    ! Catch misspelled variables
    implicit none
    integer(8), intent(in) :: iterations
    real(8), intent(in) :: kernel_mem_used
    real(8), intent(in) :: kernel_runtime, transfer_runtime

    ! NO MPI CASE - print one line
    ! Print time stats
    ! Convert runtime into same format as C using string manipulation
    write (sec_str, 63) kernel_runtime
63  format(f18.4)
    sec_str=adjustl(sec_str)
    write (us_str, 64) kernel_runtime*1000000.0_8/(1.0d+0*iterations)
64  format(f18.2)
    us_str=adjustl(us_str)
    call get_timestamp_string(timestamp)
    print '(a,a,i0,a,a,a,a,a,a,a,a,a)', trim(timestamp), &
         &" RUSHSTATS  Rush Larsen ",  iterations, " ", trim(kernel_mem_used_str), "  ", &
         & trim(sec_str), " s  ", trim(us_str), " us/iter  ", trim(variant_desc)
    flush(stdout)
  end subroutine print_runtime_stats


  ! Do sanity and consistency checks on all of m_gate. Including cross-rank if MPI mode
  ! Prints PASS or FAIL based on data check results
  ! If bad data found, will print up to 5 lines of debug info per MPI rank.
  ! Returns fail count so can be returned to caller.
  function data_check (m_gate, iterations, kernel_mem_used, nCells)
    ! Catch misspelled variables 
    implicit none
    real(8), dimension(0:), intent(inout) :: m_gate
    integer(8), intent(in) :: iterations
    real(8), intent(in) :: kernel_mem_used
    integer(8), intent(in) :: nCells
    integer(8) :: data_check ! Return value

    ! Local variables
    integer(8) :: fail_count

    ! In non-MPI mode, treat only process as rank 0
    integer :: rank = 0

    integer(8) :: i

    ! Initialize variables on every entry
    fail_count = 0

    if (rank == 0) then
       ! Print separator before and after output with function name
       call get_timestamp_string(timestamp)
       print '(a," Starting data check for sanity and consistency")', trim(timestamp)
       flush(stdout)
    end if

    ! Sanity check that kernel not giving garbage
    ! Found m_gate[0] to be ~.0.506796353074569 after 1 iteration (really 2 with warmup)
    ! and converges to 0.996321172062538 after 100 iterations.  Make sure in that bounds
    ! for now.  With a little slop (~.000001) for now (not sure rounding error expected)
    if (m_gate(0) < 0.506796) then
       call get_timestamp_string(timestamp)
       print '(a," ",a,f17.15,a,a)', trim(timestamp), &
            & "ERROR Data sanity check m_gate[0]=", m_gate(0), " < 0.506796 (0.506796353074569 min expected value) ", &
            & trim(variant_desc)
       flush(stdout)
       fail_count = fail_count + 1

    else if (m_gate(0) > 0.996322) then
       call get_timestamp_string(timestamp)
       print '(a," ",a,f17.15,a,a)', trim(timestamp), &
            & "ERROR Data sanity check m_gate[0]=", m_gate(0), " > 0.996322 (0.996321172062538 max expected value) ", &
            & trim(variant_desc)
       flush(stdout)
       fail_count = fail_count + 1
    end if

    ! Every array entry should have the same value as m_gate[0], make sure that is true
    do i=1,nCells-1
       if (m_gate(i) .ne. m_gate(0)) then
          fail_count = fail_count + 1
          ! Only print at most 5 warnings per rank
          if (fail_count < 5) then
             call get_timestamp_string(timestamp)
             print '(a," ",a,i0,a,f17.15,a,f17.15,a)', trim(timestamp), &
                  & "ERROR Data sanity check m_gate[", i, "]=", m_gate(i), " != m_gate[0]=", m_gate(0), &
                  & trim(variant_desc)
             flush(stdout)
          end if
          if (fail_count == 5) then
             call get_timestamp_string(timestamp)
             print '(a," ", a,a)', trim(timestamp), &
                  & "ERROR Data consistency check REMAINING ERROR MESSAGES SUPPRESSED! ", trim(variant_desc)
             flush(stdout)
          end if
       end if
    end do

    ! Value looks ok, check all ranks match if using MPI

    ! Print out summary PASSED or FAILED count from rank 0 only
    if (rank == 0) then
       if (fail_count == 0) then
          call get_timestamp_string(timestamp)
          print '(a,a,i0,a,a,a,f17.15,a,a)',&
               & trim(timestamp)," PASSED Data check ", iterations, " ", trim(kernel_mem_used_str), &
               & "  m_gate[0]=", m_gate(0), " ", trim(variant_desc)

          flush(stdout)
       else
          ! Convert kernel_mem_used to left justified string with leading 0
          call get_timestamp_string(timestamp)
          print '(a,a,i0,a,a,a,i0,a,f17.15,a,a)',&
               & trim(timestamp)," FAILED Data check ", iterations, " ", trim(kernel_mem_used_str), &
               & " with ", fail_count, " DATA CHECK ERRORS m_gate[0]=", m_gate(0), " ", trim(variant_desc)

          flush(stdout)
       end if
    end if

    data_check = fail_count 
  end function data_check
end function rush_larsen_gpu_omp_fort


program rush_larsen_fort
  ! Only include mpi for variants that need it
  ! Catch misspelled variables 
  implicit none
  interface 
     function rush_larsen_gpu_omp_fort(iterations, kernel_mem_used)
       integer(8),  intent(IN) :: iterations
       real(8), intent(IN) :: kernel_mem_used
       integer :: rush_larsen_gpu_omp_fort
     end function rush_larsen_gpu_omp_fort
  end interface

  ! For command line argument parsing
  character(1000) :: progName
  character(100) :: arg1char
  character(100) :: arg2char
  integer(8) :: max_iterations
  real(8) :: kernel_mem_used
  character(100) :: tag 
  integer :: rank = 0 !Rank will be 0 for the no MPI case 
  integer :: fail_count = 0

  call get_command_argument(0,progName)   !Get program name from arg 0

  !First, make sure the right number of inputs have been provided
  if(command_argument_count().ne.2) then
     if (rank == 0) then
        write(*,*) "Usage: ", trim(progName), "  Iterations  Kernel_GBs_used"
        write(*,*) " "
        write(*,*) "Measure serial baseline small:   ", trim(progName), " 100000 .00000001"
        write(*,*) "Measure serial baseline large:   ", trim(progName), "    100 10"
        write(*,*) " "
        write(*,*) trim(VERSION_STRING)
        write(*,*) " "
#if defined(COMPILERID)
        write(*,*) "VARIANT_DESC: ", "gpu_omp_fort", " [", COMPILERID, "]"
#endif /* defined(COMPILERID) */
#if !defined(COMPILERID)
        write(*,*) "VARIANT_DESC: ", "gpu_omp_fort"
#endif /* !defined(COMPILERID) */
        write(*,*) "  "
        write(*,*) "Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n"
     end if
     stop 1
  end if

  call get_command_argument(1,arg1char)   !read in the two values
  call get_command_argument(2,arg2char)

  read(arg1char,*)max_iterations                    !then, convert them to REALs
  read(arg2char,*)kernel_mem_used

  ! Don't print MPI_Init time for MPI version since the way I hid
  ! functions to enable interop makes timer routines hard to get to.

  ! Run the test
  fail_count =  rush_larsen_gpu_omp_fort (max_iterations, kernel_mem_used)

  ! Return 1 if data checks failed, before MPI Finalize
  if (fail_count .ne. 0) then
     stop 1
  end if

end program rush_larsen_fort
