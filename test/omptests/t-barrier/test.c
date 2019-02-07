
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 0; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

#define PARALLEL_A() { \
_Pragma("omp parallel num_threads(33) if (0)") \
{ \
  int tid = omp_get_thread_num(); \
  int cs = N / omp_get_num_threads(); \
  int lb = tid * cs; \
  int ub = (tid+1)*cs; \
  ub = ub > N ? N : ub; \
  for (int i = lb; i < ub; i++) { \
    A[i] = D[i]; \
  } \
  _Pragma("omp barrier") \
  double sum = 0; \
  for (int i = 1+tid; i < N; i++) { \
    sum += A[i]; \
  } \
  _Pragma("omp barrier") \
  A[tid] = sum; \
  sum = 0; \
  for (int i = 2+tid; i < N; i++) { \
    sum += A[i]; \
  } \
  _Pragma("omp barrier") \
  A[tid+1] = sum; \
  _Pragma("omp barrier") \
  B[tid] = A[tid]-A[tid+1]; \
} \
}

#define BODY_B() { \
  int tid = omp_get_thread_num(); \
  int cs = N / omp_get_num_threads(); \
  int lb = tid * cs; \
  int ub = (tid+1)*cs; \
  ub = ub > N ? N : ub; \
  for (int i = lb; i < ub; i++) { \
    A[i] = D[i]; \
  } \
  _Pragma("omp barrier") \
  double sum = 0; \
  for (int i = 1+tid; i < N; i++) { \
    sum += A[i]; \
  } \
  _Pragma("omp barrier") \
  A[tid] = sum; \
  _Pragma("omp barrier") \
  C[tid] = A[tid]-A[tid+1]-tid; \
  if (tid < omp_get_num_threads()-1) B[tid] += C[tid]; \
}

#define PARALLEL_B() { \
_Pragma("omp parallel num_threads(threads[0])") \
{ \
  BODY_B(); \
} \
}

#define PARALLEL_B5() { PARALLEL_B() PARALLEL_B() PARALLEL_B() PARALLEL_B() PARALLEL_B() }

#define BODY_NP() { \
_Pragma("omp parallel num_threads(16)") { \
  int b = omp_get_thread_num()*16; \
_Pragma("omp parallel num_threads(16)") { \
  int tid = omp_get_thread_num(); \
  int cs = N / omp_get_num_threads(); \
  int lb = tid * cs; \
  int ub = (tid+1)*cs; \
  ub = ub > N ? N : ub; \
  for (int i = lb; i < ub; i++) { \
    A[i] = D[i]; \
  } \
  _Pragma("omp barrier") \
  double sum = 0; \
  for (int i = 1+tid; i < N; i++) { \
    sum += A[i]; \
  } \
  _Pragma("omp barrier") \
  A[tid] = sum; \
  _Pragma("omp barrier") \
  C[tid] = A[tid]-A[tid+1]-tid; \
  if (tid < omp_get_num_threads()-1) B[b+tid] += C[tid]; else B[b+tid]=0; \
} \
} \
}

int main(void) {
  check_offloading();

  double A[N+2], B[N+2], C[N+2], D[N+2], E[N+2];

  INIT();

  long cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int gpu_threads = 768;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  //
  // Test: Barrier in a serialized parallel region.
  //
  TESTD("omp target teams num_teams(1) thread_limit(33)", {
    PARALLEL_A()
  }, VERIFY(0, 1, B[i], i+1));

  //
  // Test: Barrier in a parallel region.
  //
  for (int t = 1; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    TESTD("omp target teams num_teams(1) thread_limit(max_threads)", {
    ZERO(B);
    PARALLEL_B5()
    }, VERIFY(0, threads[0]-1, B[i], 5));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: Barrier in consecutive parallel regions with variable # of threads.
  //
  TESTD("omp target teams num_teams(1) thread_limit(max_threads)", {
  ZERO(B);
  for (int t = 2; t <= max_threads; t++) {
    int threads[1]; threads[0] = t;
    PARALLEL_B()
  }
  }, VERIFY(0, max_threads-1, B[i], max_threads-i-1));

  //
  // Test: Single thread in target region.
  //
  TESTD("#pragma omp target", {
  ZERO(B);
  BODY_B()
  }, VERIFY(0, 1, C[i], 491535));

  //
  // Test: Barrier in target parallel.
  //
  for (int t = 2; t <= max_threads; t++) {
    ZERO(B);
    int threads; threads = t;
    TESTD("omp target parallel num_threads(threads)", {
      BODY_B();
    }, VERIFY(0, t-1, B[i], (trial+1)*1));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: Barrier in nested parallel in target region.
  //
  if (!cpuExec) {
    ZERO(B);
    TEST({
      BODY_NP();
    }, VERIFY(0, 16*16, B[i], (i > 0 && (i+1) % 16 == 0 ? 0 : (trial+1)*1)) );
  } else {
    DUMP_SUCCESS(1);
  }

  // target parallel + parallel
  // target + simd
  // target/teams/parallel with varying numbers of threads

}
