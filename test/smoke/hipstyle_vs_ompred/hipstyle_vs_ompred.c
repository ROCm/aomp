//
//  hipstyle_vs_ompred.c: This test shows how a hipstyle reduction in an openmp
//                   target region compares to the simple-to-code omp reduction.
//
#include <omp.h>
#include <stdio.h>

#define N 5000001

//  These macros allows compilation with -DNUM_TEAMS=<testval> and
//  -DNUM_THREADS=<testval> Default NUM_TEAMS set for vega number of CUs
#ifndef NUM_TEAMS
#define NUM_TEAMS 60
#endif
#ifndef NUM_THREADS
#define NUM_THREADS 1024
#endif

void __kmpc_barrier(void *Loc, int TId);

int main() {
  int main_rc = 0;
  double expect = (double)(((double)N - 1) * (double)N) / 2.0;

  // dry runs to initialize hsa_queue's
  for(int i = 0; i < 4; i++) {
    #pragma omp target
    {}
  }

  // Initialize GPUs with a simple kernel
  #pragma omp target
    printf("GPUs initialized NUM_TEAMS:%d NUM_THREADS:%d\n",
		    NUM_TEAMS, NUM_THREADS);

  // ---------  Calculate sum using manual reduction technique -------

  double hipstyle_sum = 0.0;
  double t0 = omp_get_wtime();
  #pragma omp target teams distribute parallel for num_teams(NUM_TEAMS)        \
    num_threads(NUM_THREADS) map(tofrom: hipstyle_sum)
  for (int kk = 0; kk < NUM_TEAMS * NUM_THREADS; kk++) {
    // A HIP or CUDA kernel will use builtin values with names like these
    // We get these values from the OpenMP target API;
    unsigned int BlockIdx_x  = omp_get_team_num();
    unsigned int ThreadIdx_x = omp_get_thread_num();
    unsigned int GridDim_x   = NUM_TEAMS;   // could be omp_get_num_teams()
    unsigned int BlockDim_x  = NUM_THREADS; // could be omp_get_num_threads()

    // tb_sum is an LDS array that is shared only within a team.
    // The openmp pteam allocator for shared arrays does not work yet.
    // But this attribute makes the array LDS.
    static __attribute__((address_space(3))) double tb_sum[NUM_THREADS];

    int i = BlockDim_x * BlockIdx_x + ThreadIdx_x;
    tb_sum[ThreadIdx_x] = 0.0;
    for (; i < N; i += BlockDim_x * GridDim_x)
      tb_sum[ThreadIdx_x] += (double)i;

    // clang does not permit #pragma omp barrier here
    // But we need one, so use the internal libomptarget barrier
    #if defined(__AMDGCN__) || defined(__NVPTX__)
    __kmpc_barrier(NULL, ThreadIdx_x);
    #endif

    // Reduce each team into tb_sum[0]
    for (int offset = BlockDim_x / 2; offset > 0; offset /= 2) {
      if (ThreadIdx_x < offset)
        tb_sum[ThreadIdx_x] += tb_sum[ThreadIdx_x + offset];
      #if defined(__AMDGCN__) || defined(__NVPTX__)
      __kmpc_barrier(NULL, ThreadIdx_x);
      #endif
    }

    // Atomically reduce each teams sum to a single value.
    // This is concurrent access by NUM_TEAMS workgroups to a single global val
    // For machines with hardware fp atomic use the hint here.
    if (ThreadIdx_x == 0) {
      #pragma omp atomic
      hipstyle_sum += tb_sum[0];
    }
    // FYI. In a real code, if reduced value (hipstyle_sum) were needed on GPU
    // after this point you would need some sort of cross-team barrier.
  } // END TARGET REGION

  double t1 = omp_get_wtime() - t0;
  if (hipstyle_sum == expect) {
    printf("Success HIP-style     sum of %d integers is: %14.0f in %f secs\n",
           N - 1, hipstyle_sum, t1);
  } else {
    printf("FAIL HIPSTYLE SUM N:%d result: %f != expect: %f \n", N - 1,
           hipstyle_sum, expect);
    main_rc = 1;
  }

  // ---------  Calculate sum using OpenMP reduction -------

  double ompred_sum = 0.0;
  double t2 = omp_get_wtime();
  #pragma omp target teams distribute parallel for       \
    map(tofrom: ompred_sum) reduction(+:ompred_sum)
  for (int ii = 0; ii < N; ++ii)
    ompred_sum += (double)ii;

  double t3 = omp_get_wtime() - t2;
  if (ompred_sum == expect) {
    printf("Success OMP reduction sum of %d integers is: %14.0f in %f secs\n",
           N - 1, ompred_sum, t3);
  } else {
    printf("FAIL REDUCTION SUM N:%d result: %f != expect: %f \n", N - 1,
           ompred_sum, expect);
    main_rc = 1;
  }
  return main_rc;
}
