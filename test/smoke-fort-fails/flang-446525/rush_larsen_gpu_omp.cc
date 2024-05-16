/*
  Copyright (c) 2019-21, Lawrence Livermore National Security, LLC. and other
  Goulash project contributors LLNL-CODE-795383, All rights reserved.
  For details about use and distribution, please read LICENSE and NOTICE from
  the Goulash project repository: http://github.com/llnl/goulash
  SPDX-License-Identifier: BSD-3-Clause
*/

/* Designed to allow direct performance comparisons between 
 * naively written HIP/CUDA and OpenMP GPU offloading schemes
 * in a variety of coding styles and languages of a 
 * parameterized embarrassingly parallel Rush Larsen kernel. 
 * Also allows testing build systems (including SPACK) handling
 * of complicated build situations that LLNL cares about.
 *
 * Designed to create several single file test variants 
 * where no -D options required to select the variant
 * and no include files are needed.
 * Almost all code in this file is identical between 
 * variants (this is intentional).
 * MPI support is ifdefed out for non-MPI variants.
 * 
 * The key computational kernel can be located by
 * searching for: KERNEL
 * 
 * Designed to create trivial variants of the same
 * test to be compiled with different compilers for
 * the interoperability tests.   This is why most
 * functions are static and the main kernel test
 * is called rush_larsen_gpu_omp (preprocessor
 * defines this and VARIANT_DESC based on 
 * variant configuration).
 *
 * The naming convention of the variant copies is
 * intended to indicate variant abilities:
 * _cpu_serial      - single threaded, no OpenMP, on CPU
 * _cpu_omp         - use OpenMP to spawn threads on CPU
 * _gpu_omp         - uses OpenMP to offload to GPU 
 * _gpu_hip         - uses HIP to offload to AMD or Nvidia GPU
 * _gpu_lambda_hip  - RAJA-like lambda HIP variant
 * _gpu_cuda        - uses CUDA to offload to Nvidia GPU
 * _gpu_lambda_cuda - RAJA-like lambda CUDA variant
 * *_mpi            - uses and exercises MPI e.g. _gpu_omp_mpi 
 * *_fort           - Fortran version e.g. _gpu_omp_mpi_fort
 *
 * For the interop tests, there is an additional suffix
 * to indicate different copies of the same configuration
 * that are intended to be compiled by different compilers:
 * _compiler1  - E.g., rush_larsen_gpu_omp_compiler1.cc
 *
 * VARIANT_DESC set by preprocessor directives to
 * the configuration of this file.
 *
 * Recommended that a -DCOMPILERID be set to the compiler used to compile each file:
 *
 * /opt/rocm-4.0.1/llvm/bin/clang++  -o rush_larsen_cpu_omp -O3 -g "-DCOMPILERID=rocm-4.0.1" -fopenmp rush_larsen_cpu_omp.cc
 * 
 * Run with no arguments for suggested arguments, for example:
 *   Usage: ./rush_larsen_cpu_omp  Iterations  Kernel_GBs_used
 *
 *     Measure serial launch overhead:   env OMP_NUM_THREADS=1 ./rush_larsen_cpu_omp 100000 .00000001
 *     Measure thread launch overhead:   ./rush_larsen_cpu_omp 100000 .00000001
 *     Measure kernel performance: ./rush_larsen_cpu_omp    100 10
 * 
 * 
 * The Goulash project conceived of and designed by David Richards, 
 * Tom Scogland, and John Gyllenhaal at LLNL Oct 2019.
 * Please contact John Gyllenhaal (gyllenhaal1@llnl.gov) with questions.
 *
 * Rush Larsen core CUDA/OpenMP kernels written by Rob Blake (LLNL) Sept 2016.
 * The goulash Rush Larsen tests add benchmarking infrastructure 
 * around this incredibly useful compact GPU test kernel.   Thank you Rob!
 * 
 * Inline performance measurements added (nvprof not needed)
 * by John Gyllenhaal at LLNL 11/10/20.
 *
 * Command line argument handling, performance difference printing in
 * form easily found with grep, OpenMP thread mapping and initial data
 * sanity checks on just the first array element calculated by kernel
 * by John Gyllenhaal at LLNL 03/22/21
 * 
 * Pulled code from print_openmp_mapping.c by John Gyllenhaal at
 * LLNL written June 2020 which was based on mpibind tests
 * (https://github.com/LLNL/mpibind) by Edgar Leon at LLNL
 *
 * RAJA-perf-suite-like (https://github.com/LLNL/RAJAPerf)
 * C++ lambda versions created by Jason Burmark at LLNL 06/16/21
 * 
 * C-like Fortran ports (by hand) of C++ version of 
 * rush larsen variants by John Gyllenhaal at LLNL 06/28/21.
 * 
 * MPI stat aggregation and MPI exercisers written
 * by John Gyllenhaal at LLNL based on previous user issue reproducers.
 * Pulled into rush larsen tests on 07/02/21 by John Gyllenhaal at LLNL.
 *
 * Enhanced data checks of all kernel generated array data, including 
 * across MPI ranks by John Gyllenhaal at LLNL 07/03/21
 *
 * Interop versions create by John Gyllenhaal at LLNL 07/14/21
 * to test mixing all the Rush Larsen tests with multiple GPU compilers
 * all in one final executable.
 * 
 * Initial test generator from template files, including Makefiles
 * created by John Gyllenhaal at LLNL 07/21/21 for V2.0RC1
 *
 * V2.0RC1 07/21/21 Added MPI support, interop version, enhanced data checks.
 * V1.2 06/28/21 Fortran and C++ lambda versions added, consistent use of long type
 * V1.1 03/22/21 command line args, perf diffs, maps, checks return codes and answer
 * V1.0 11/10/20 initial release, hard coded inputs, no error checking
 */

/* Allow version to be printed in output */
#define VERSION_STRING "Version 2.0 RC1 (7/21/21)"

/* If NO_STATIC defined, make all support routines non-static (visible) */
#define STATIC static

/* Preprocessor macro rushglue(x,y) glues two defined values together */
#define rushglue2(x,y) x##y
#define rushglue(x,y) rushglue2(x,y) 

/* Preprocessor macro rushxstr(s) converts value to string */
#define rushxstr2(s) #s
#define rushxstr(s) rushxstr2(s)

/* For interop version, can #define VARIANT_ID 1, etc. to make different named kernels.
 * If not defined, make empty string 
 */
#ifndef VARIANT_ID
#define VARIANT_ID
#endif

/* Create unique gpu_omp based on file #def and #undef
 * settings that is used to create rush_larsen function
 * call name and to annotate key lines of output
 */
#undef TARGET_TAG
#define TARGET_TAG gpu_omp

/* Append _mpi to indicate using MPI */
#undef RUSH_MPI_TAG
#define RUSH_MPI_TAG TARGET_TAG

/* Append VARIANT_ID (may be empty) to get gpu_omp */
#undef gpu_omp
#define CXX_VARIANT_TAG rushglue(RUSH_MPI_TAG,VARIANT_ID)


/* Generate VARIANT_DESC string that annotates the end of key output
 * lines spread across this whole file.  Uses C trick that
 * "omp" " [" "g++" "]" 
 * is equivalent to
 * "omp [g++]"
 * Since I could not figure out how to create one big string
 * with the preprocessor.
 */
#ifdef COMPILERID
#define VARIANT_DESC rushxstr(gpu_omp) " ["  rushxstr(COMPILERID) "]"
#else
#define VARIANT_DESC rushxstr(gpu_omp)
#endif

#include <omp.h>
#include <cstdlib>
#include <cmath>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <sched.h>

/* Get raw time in seconds as double (a large number).
 * Returns -1.0 on unexpected error.
 */
STATIC double get_raw_secs( void )
{
    struct timeval ts;
    int status;
    double raw_time;
        
    /* Get wall-clock time */
    /* status = getclock( CLOCK_REALTIME, &ts ); */
    status = gettimeofday( &ts, NULL );
        
    /* Return -1.0 on error */
    if( status != 0 ) return -1.0;
        
    /* Convert structure to double (in seconds ) (a large number) */
    raw_time = (double)ts.tv_sec + (double)ts.tv_usec * 1e-6;

    return (raw_time);
}
        
/* Returns base time.  If new_time >= 0, 
 * sets base_time to new_time before returning.
 * Using this as access method to static variable
 * in a way I can trivially emulate in fortran.
 *
 * Note: Lock shouldn't be needed, since even if multiple
 *       threads initialize this, it will be to basically
 *       the same value.
 */
STATIC double get_base_time(double new_time)
{
    static double base_time = -1.0;

    /* If passed value >= 0, use as new base_time */ 
    if (new_time >= 0.0)
        base_time = new_time;

    return(base_time);
}

/* Returns time in seconds (double) since the first call to secs_elapsed
 * (i.e., the first call returns 0.0).
 */
STATIC double secs_elapsed( void )
{
    double new_time;
    double base_time;
        
    /* Get current raw time (a big number) */
    new_time = get_raw_secs();
        
    /* Get the offset since first time called (pass -1 to query)*/
    base_time = get_base_time(-1.0);
    
    /* If base time not set (negative), set to current time (pass in positive secs)*/
    if (base_time < 0.0)
        base_time = get_base_time(new_time);
     
    /* Returned offset from first time called */
    return (new_time - base_time);
}

/* Works like vfprintf, except prefixes wall-clock time (using secs_elapsed)
 * and the difference since last vfprintf.
 * Also flushes out after printing so messages appear immediately
 * Used to implement printf_timestamp, punt, etc.
 */
STATIC double last_printf_timestamp=0.0;
STATIC void vfprintf_timestamp(FILE*out, const char * fmt, va_list args)
{
    char buf[4096];
    int rank = -1;  /* Don't print rank for serial runs */

    /* Get wall-clock time since first call to secs_elapsed */
    double sec = secs_elapsed();
    double diff = sec - last_printf_timestamp;
    last_printf_timestamp=sec;

    /* Print out passed message to big buffer*/
    vsnprintf(buf, sizeof(buf), fmt, args);

    /* No MPI case */
    if (rank < 0)
    {
        /* Print out timestamp and diff seconds with buffer*/
        fprintf(out, "%7.3f (%05.3fs): %s", sec, diff, buf);
    }
    /* MPI case, add rank */
    else
    {
        /* Print out timestamp and diff seconds and MPI rank with buffer*/
        fprintf(out, "%3i: %7.3f (%05.3fs): %s", rank, sec, diff, buf);
    }

    /* Flush out, so message appears immediately */
    fflush(out);
}

/* Prints to stdout for all MPI ranks with timestamps and time diffs */
STATIC void printf_timestamp(const char * fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    /* Use helper routine to actually do print and flush */
    vfprintf_timestamp(stdout,fmt,args);

    va_end(args);
}

/* Prints to stdout for only Rank 0 with timestamps and time diffs.
 * For all other MPI ranks, the message is thrown away.
 */
STATIC void rank0_printf_timestamp(const char * fmt, ...)
{
    int rank = 0;  /* Non-mpi runs always get printed out */
    va_list args;
    va_start(args, fmt);


    /* Only print if rank 0 (or non-MPI program) */
    if (rank == 0)
    {
        /* Use helper routine to actually do print and flush */
        vfprintf_timestamp(stdout,fmt,args);
    }

    va_end(args);
}

/* Prints to stderr (flushes stdout first) with timestamp and exits */
STATIC void punt(const char * fmt, ...)
{
    va_list args;
    va_start(args, fmt);

    /* Flush stdout, so pending message appears before punt message */
    fflush(stdout);

    /* Use helper routine to actually do print and flush */
    vfprintf_timestamp(stderr,fmt,args);

    va_end(args);

    /* Abort the program */
    exit(1);
}

/* The maximum number of threads supported */
#define MAX_SIZE 1024


/* Print kernel runtime stats and aggregate across MPI processes if necessary.
 * Prints one liner if not using MPI
 */
STATIC void print_runtime_stats(long iterations, double kernel_mem_used, double kernel_runtime, double transfer_runtime)
{
    /* NO MPI CASE - print one line */
    /* Print time stats */
    printf_timestamp("RUSHSTATS  Rush Larsen %li %.8f  %.4lf s  %.2lf us/iter  %.3f s datatrans %s\n",  
                     iterations, kernel_mem_used, kernel_runtime, (double)(kernel_runtime)*1000000.0/(double) iterations, transfer_runtime, VARIANT_DESC);
}

/* Do sanity and consistency checks on all of m_gate. Including cross-rank if MPI mode 
 * Prints PASS or FAIL based on data check results
 * If bad data found, will print up to 5 lines of debug info per MPI rank.
 * Returns fail count so can be returned to caller.
 */
STATIC long data_check(double *m_gate, long iterations, double kernel_mem_used, long nCells)
{
    /* If in MPI mode, in order to prevent MPI hangs on data fails, 
     * need to do MPI allreduces even if earlier checks fail.   
     * As a bonus, this algorithm allows rank 0 to be able to 
     * print out how many data check failures occurred across all ranks.
     */
    long fail_count = 0;


    rank0_printf_timestamp("Starting data check for sanity and consistency\n");

    /* Sanity check that calculation not giving garbage
     * Found m_gate[0] to be ~.0.506796353074569 after 1 iteration (really 2 with warmup)
     * and converges to 0.996321172062538 after 100 iterations.  Make sure in that bounds
     * for now.  With a little slop (~.000001) for now (not sure rounding error expected)
     */
    if (m_gate[0] < 0.506796) 
    {
        printf_timestamp("ERROR Data sanity check m_gate[0]=%.15lf < 0.506796 (0.506796353074569 min expected value) %s\n", m_gate[0], VARIANT_DESC);
        fail_count++;
    }

    if (m_gate[0] > 0.996322)
    {
        printf_timestamp("ERROR Data sanity check m_gate[0]=%.15lf > 0.996322 (0.996321172062538 max expected value) %s\n", m_gate[0], VARIANT_DESC);
        fail_count++;

    }

    /* Every array entry should have the same value as m_gate[0], make sure that is true */
    for (long i = 1; i < nCells; i++)
    {
        if (m_gate[i] != m_gate[0])
        {
            fail_count++;
            /* Only print at most 5 warnings per rank */
            if (fail_count < 5)
            {
                printf_timestamp("ERROR Data consistency check m_gate[%i]=%.15lf != m_gate[0]=%.15lf %s\n", 
                                 i, m_gate[i], m_gate[0], VARIANT_DESC);
            }
            if (fail_count == 5)
            {
                printf_timestamp("ERROR Data consistency check REMAINING ERROR MESSAGES SUPPRESSED! %s\n", VARIANT_DESC);
            }

        }
    }

    /* Value looks ok, check all ranks match if using MPI */

    /* Print out summary PASSED or FAILED count from rank 0 only*/
    if (fail_count == 0)
    {
        rank0_printf_timestamp("PASSED Data check %ld %.8f  m_gate[0]=%.15lf %s\n", iterations, kernel_mem_used, m_gate[0], VARIANT_DESC);
    }
    else
    {
        rank0_printf_timestamp("FAILED Data check %ld %.8f  with %li DATA CHECK ERRORS m_gate[0]=%.15lf %s\n", 
                               iterations, kernel_mem_used, fail_count, m_gate[0], VARIANT_DESC);
    }

    /* Return the number of data_check failures detected (across all ranks, if MPI mode) */
    return (fail_count);
}


/* If using OpenMP offloading, make sure GPU works before doing test */
STATIC void verify_gpu_openmp(int gpu_id)
{
    /* If using GPU, make sure GPU OpenMP gpu offloading works before doing test */
    int runningOnGPU = 0;

    char mpi_desc[50]="";


    rank0_printf_timestamp("Selecting GPU %i as default device%s\n", gpu_id, mpi_desc); 

    /* Pick GPU to use to exercise selection call */
    omp_set_default_device(gpu_id);


    rank0_printf_timestamp("Launching OpenMP GPU test kernel%s\n", mpi_desc); 

    /* Test if GPU is available using OpenMP4.5 legal code */
#pragma omp target map(from:runningOnGPU)
    {
        if (omp_is_initial_device() == 0)
            runningOnGPU = 1;
    }

    /* If still running on CPU, GPU must not be available, punt */
    if (runningOnGPU != 1)
        punt("ERROR: OpenMP GPU test kernel did NOT run on GPU %i %s", gpu_id, VARIANT_DESC);

    rank0_printf_timestamp("Verified OpenMP target test kernel ran on GPU%s\n", mpi_desc);
}

/* Returns secs_elapsed after MPI barrier(if MPI) and printing desc to rank 0 */
STATIC double sync_starttime(const char *desc)
{
    double start_time;

    rank0_printf_timestamp("%s", desc);

    start_time=secs_elapsed();
    return (start_time);
}

/* Returns secs_elapsed before MPI barrier (if MPI) and printing desc to rank 0 */
STATIC double sync_endtime(const char *desc)
{
    double end_time;

    end_time=secs_elapsed();

    rank0_printf_timestamp("%s", desc);

    return (end_time);
}

/* Sets up and runs the doRushLarsen kernel 'iterations' times, 
 * allocating CPU arrays and perhaps GPU arrays to consume 
 * kernel_mem_used GBs of memory.   
 *
 * This polynomial is a fit to the dynamics of a small part of a cardiac
 * myocyte, specifically the fast sodium m-gate described here:
 * https://www.ncbi.nlm.nih.gov/pubmed/16565318
 *
 * Does exactly the same work on every cell.   Can scale from one cell
 * to filling entire memory.   Does use cell's value as input
 * to calculations.
 * 
 * Returns number of data check failures, returns 0 if all data checks out.
 */
extern "C" long rush_larsen_gpu_omp(long iterations, double kernel_mem_used)
{
    double kernel_starttime,kernel_endtime, kernel_runtime;
    double transfer_starttime,transfer_endtime, transfer_runtime;
    long nCells;
    long status_point;
    long fail_count =0;

    /* To make interop performance easier to compare,
     * start this file's timers over every time called.
     * Reset this file's secs_elapsed() counter to 0 
     */
    get_base_time(get_raw_secs());

    /* Synchronize printf timestamps across MPI ranks */
    last_printf_timestamp= secs_elapsed();

    /* Print separator before and after output with function name*/
    rank0_printf_timestamp("--------------- Begin rush_larsen_" VARIANT_DESC " (timer zeroed) ---------------\n");

    /* For print niceness, make .00000001 lower bound on GB memory */
    if (kernel_mem_used < .00000001)
        kernel_mem_used = .00000001;

    /* Calculate nCells from target memory target */
    nCells = (long) ((kernel_mem_used * 1024.0 * 1024.0 * 1024.0) / (sizeof(double) * 2));

    /* Must have at least 1 cell */
    if (nCells < 1)
        nCells = 1;

    /* Must have at least 1 iteration */
    if (iterations < 1)
        iterations=1;

    /* Give status every 10% of iterations */
    status_point=iterations/10;
    /* Must be at least 1 to make mod work*/
    if (status_point < 1)
        status_point = 1;
            
    /* Print what we are running */
    rank0_printf_timestamp("START Rush Larsen %ld %.8f  cells %ld  %s\n", iterations, kernel_mem_used, nCells, VARIANT_DESC); 
    rank0_printf_timestamp("%s\n", VERSION_STRING); 

    /* If using GPU, make sure GPU works before doing test */
    verify_gpu_openmp(0);

    rank0_printf_timestamp("Allocating and initializing kernel arrays\n");

    double* m_gate = (double*)calloc(nCells,sizeof(double));
    if (m_gate == NULL)
    {
        punt("%s failed calloc m_gate",VARIANT_DESC);
    }
         
    double* Vm = (double*)calloc(nCells,sizeof(double));
    if (Vm == NULL)
    {
        punt("%s failed calloc Vm", VARIANT_DESC);
    }

    /* No data transfer time if not using GPU */
    transfer_starttime=0.0;
    transfer_endtime=0.0;

    transfer_starttime=sync_starttime("Starting omp data map of CPU arrays to GPU\n");
#pragma omp target enter data map(to: m_gate[:nCells])
#pragma omp target enter data map(to: Vm[:nCells])
    transfer_endtime=sync_endtime("Finished omp data map of CPU arrays to GPU\n");
    transfer_runtime=transfer_endtime-transfer_starttime;
            
    /* Do the iterations asked for plus 1 for warmup */
    for (long itime=0; itime<=iterations; itime++) 
    {
        /* Print warmup message for 0th iteration */
        if (itime == 0) 
        {
            rank0_printf_timestamp("Launching warmup iteration (not included in kernel timings)\n");
        }
                
        /* Print status every 10% of iterations */
        else if (((itime-1) % status_point) == 0)
        {   
            if (itime==1)
            {

            }

            if (itime == 1)
            {
                rank0_printf_timestamp("Starting kernel timings for Rush Larsen %ld %.8f\n", iterations, kernel_mem_used);
            }

            rank0_printf_timestamp("Starting iteration %6li\n", itime);
        }

        /* Start timer after warm-up iteration 0 */
        if (itime==1)
        {
            kernel_starttime=secs_elapsed();
        }
                
        /*
         * RUSH LARSEN KERNEL BEING TIMED START
         */
#pragma omp target teams distribute parallel for 
        for (long ii=0; ii<nCells; ii++) 
        {
            double sum1,sum2;
            const double x = Vm[ii];
            const int Mhu_l = 10;
            const int Mhu_m = 5;
            const double Mhu_a[] = { 9.9632117206253790e-01,  4.0825738726469545e-02,  6.3401613233199589e-04,  4.4158436861700431e-06,  1.1622058324043520e-08,  1.0000000000000000e+00,  4.0568375699663400e-02,  6.4216825832642788e-04,  4.2661664422410096e-06,  1.3559930396321903e-08, -1.3573468728873069e-11, -4.2594802366702580e-13,  7.6779952208246166e-15,  1.4260675804433780e-16, -2.6656212072499249e-18};

            sum1 = 0;
            for (int j = Mhu_m-1; j >= 0; j--)
                sum1 = Mhu_a[j] + x*sum1;
            sum2 = 0;
            int k = Mhu_m + Mhu_l - 1;
            for (int j = k; j >= Mhu_m; j--)
                sum2 = Mhu_a[j] + x * sum2;
            double mhu = sum1/sum2;

            const int Tau_m = 18;
            const double Tau_a[] = {1.7765862602413648e+01*0.02,  5.0010202770602419e-02*0.02, -7.8002064070783474e-04*0.02, -6.9399661775931530e-05*0.02,  1.6936588308244311e-06*0.02,  5.4629017090963798e-07*0.02, -1.3805420990037933e-08*0.02, -8.0678945216155694e-10*0.02,  1.6209833004622630e-11*0.02,  6.5130101230170358e-13*0.02, -6.9931705949674988e-15*0.02, -3.1161210504114690e-16*0.02,  5.0166191902609083e-19*0.02,  7.8608831661430381e-20*0.02,  4.3936315597226053e-22*0.02, -7.0535966258003289e-24*0.02, -9.0473475495087118e-26*0.02, -2.9878427692323621e-28*0.02,  1.0000000000000000e+00};

            sum1 = 0;
            for (int j = Tau_m-1; j >= 0; j--)
                sum1 = Tau_a[j] + x*sum1;
            double tauR = sum1;
            m_gate[ii] += (mhu - m_gate[ii])*(1-exp(-tauR));
        }; /* This ';' required for lambda variants */
        /*
         * RUSH LARSEN KERNEL BEING TIMED END
         */
    }

    /* Get time after all iterations */
    kernel_endtime=secs_elapsed();

    /* Calculate kernel runtime */
    kernel_runtime = kernel_endtime-kernel_starttime;

    rank0_printf_timestamp("Finished kernel timings for Rush Larsen %ld %.8f\n", iterations, kernel_mem_used);

    /* Print kernel runtime stats, syncs and aggregates MPI rank (if MPI mode) */
    print_runtime_stats(iterations, kernel_mem_used, kernel_runtime, transfer_runtime);

    /* Transfer GPU m_gate kernel memory to CPU kernel memory for data checks */
    rank0_printf_timestamp("Starting omp target update of GPU result array to CPU array\n");
#pragma omp target update from (m_gate[0:nCells])
    rank0_printf_timestamp("Finished omp target update of GPU result array to CPU array\n");

    /* Do sanity and consistency checks on all of m_gate. Including cross-rank if in MPI mode.
     * Prints PASS or FAIL based on data check results
     * Returns fail count so can be returned to caller.
     */
    fail_count = data_check(m_gate, iterations, kernel_mem_used, nCells);

    /* Free kernel GPU Memory */
#pragma omp target exit data map(delete:m_gate[:nCells])
#pragma omp target exit data map(delete:Vm[:nCells])

    /* Free kernel CPU Memory */
    free(Vm);
    free(m_gate);

    rank0_printf_timestamp("DONE Freed memory %s\n", VARIANT_DESC);

    /* Print separator before and after output */
    rank0_printf_timestamp("----------------- End rush_larsen_" VARIANT_DESC " ---------------\n");
   
    /* Return number of data check failures */
    return (fail_count);
}

/* Main driver (single test) when not being used in interop test */
int main(int argc, char* argv[]) 
{
    long max_iterations=1;
    double kernel_mem_used=0.0;
    int rank = 0; /* Rank will be 0 for the no MPI case */
    int fail_count = 0;

    if (argc != 3)
    {
        if (rank == 0)
        {
            printf("Usage: %s  Iterations  Kernel_GBs_used\n", argv[0]);
            printf("\n");
            printf("Measure serial baseline small: %s 100000 .00000001\n", argv[0]);
            printf("Measure serial baseline large: %s    100 10\n", argv[0]);
            printf("\n");
            printf("%s\n", VERSION_STRING);
            printf("\n");
            printf("RUSH LARSEN VARIANT:  rush_larsen_gpu_omp\n");
            printf("VARIANT_DESC: " VARIANT_DESC "\n");
            printf("\n");
            printf("Questions? Contact John Gyllenhaal (gyllenhaal1@llnl.gov)\n");
        }
        exit (1);
    }

    /* Get iteration count and target kernel memory used arguments */
    max_iterations = atol(argv[1]);
    kernel_mem_used=atof(argv[2]);

    /*
     * Do the rush larsen test with the specified configuration 
     */
    fail_count = rush_larsen_gpu_omp(max_iterations, kernel_mem_used);

    /* Return 1 if data checks failed */
    if (fail_count == 0)
        return(0);
    else
        return(1);
}
