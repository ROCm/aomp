
#include <stdio.h>
#include <omp.h>
#include <math.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  INIT();

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int gpu_threads = 128;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  //
  // Test: omp_get_thread_num()
  //
  ZERO(A);
  TESTD("omp target parallel num_threads(max_threads)", {
    int tid = omp_get_thread_num();
    A[tid] += tid;
  }, VERIFY(0,  max_threads, A[i], i*(trial+1)));

  //
  // Test: Execute parallel on device
  //
  TESTD("omp target parallel num_threads(max_threads)", {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] = D[j] + E[j];
      }
    }, VERIFY(0, max_threads*4, B[i], (double)0));

  //
  // Test: if clause serial execution of parallel region on host
  //
  ZERO(A);
  TESTD("omp target parallel num_threads(max_threads) if(0)", {
      int tid = omp_get_thread_num();
      A[tid] = tid;
  }, VERIFY(0, max_threads, A[i], 0));

  //
  // Test: if clause parallel execution of parallel region on device
  //
  ZERO(A);
  TESTD("omp target parallel num_threads(max_threads) if(A[0] == 0)", {
      int tid = omp_get_thread_num();
      A[tid] = tid + omp_is_initial_device();
  }, VERIFY(0, max_threads, A[i], i + cpuExec));

  //
  // Test: if clause serial execution of parallel region on device
  //
  ZERO(A);
  TESTD("omp target parallel num_threads(max_threads) if(parallel: 0)", {
      int tid = omp_get_thread_num();
      A[tid] = !omp_is_initial_device();
  }, VERIFY(0, max_threads, A[i], i == 0 ? 1 - cpuExec : 0));


  //
  // Test: if clause parallel execution of parallel region on host
  //
  ZERO(A);
  TESTD("omp target parallel num_threads(max_threads) if(target: 0) if(parallel: A[0] == 0)", {
      int tid = omp_get_thread_num();
      A[tid] = tid + omp_is_initial_device();
  }, VERIFY(0, /* bound to */ cpu_threads, A[i], i+1));


  //
  // Test: if clause serial execution of parallel region on device without num_threads clause
  //
  ZERO(A);
  TESTD("omp target parallel if(parallel: A[0] > 0)", {
      int tid = omp_get_thread_num();
      A[tid] = omp_get_num_threads();
  }, VERIFY(0, 1, A[0], 1));

  //
  // Test: if clause parallel execution of parallel region on device without num_threads clause
  //       The testcase should be launched with the default number of threads.
  //
  ZERO(A);
  #pragma omp target parallel
  {
    // Get default number of threads launched by this runtime.
    B[0] = omp_get_num_threads();
  }
  TESTD("omp target parallel if(parallel: A[0] == 0)", {
      int tid = omp_get_thread_num();
      A[tid] = omp_get_num_threads();
  }, VERIFY(0, 1, A[0], B[0]));

  //
  // Test: if clause parallel execution of parallel region on device with num_threads clause
  //
  ZERO(A);
  TESTD("omp target parallel num_threads(2) if(parallel: A[0] == 0)", {
      int tid = omp_get_thread_num();
      A[tid] = omp_get_num_threads();
  }, VERIFY(0, 1, A[0], 2));

  //
  // Test: proc_bind clause
  //
  TESTD("omp target parallel num_threads(max_threads) proc_bind(master)", {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] = 1 + D[j] + E[j];
      }
  }, VERIFY(0, max_threads*4, B[i], 1));
  TESTD("omp target parallel num_threads(max_threads) proc_bind(close)", {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] = 1 + D[j] + E[j];
      }
  }, VERIFY(0, max_threads*4, B[i], 1));
  TESTD("omp target parallel num_threads(max_threads) proc_bind(spread)", {
      int i = omp_get_thread_num()*4;
      for (int j = i; j < i + 4; j++) {
        B[j] = 1 + D[j] + E[j];
      }
  }, VERIFY(0, max_threads*4, B[i], 1));

  //
  // Test: num_threads on parallel.
  //
  for (int t = 1; t <= max_threads; t++) {
    ZERO(A);
    int threads[1]; threads[0] = t;
    TESTD("omp target parallel num_threads(threads[0])", {
        int tid = omp_get_thread_num();
        A[tid] = 99;
    }, VERIFY(0, 128, A[i], 99*(i < t)));
  }
  DUMP_SUCCESS(gpu_threads-max_threads);

  //
  // Test: sharing of variables from host to parallel region.
  //
  ZERO(A);
  {
    double tmp = 1;
    A[0] = tmp;
    TESTD("omp target parallel map(tofrom: tmp) num_threads(1)", {
        tmp = 2;
        A[0] += tmp;
    }, VERIFY(0, 1, A[i]+tmp, (1+trial)*2+1+2));
  }

  //
  // Test: private clause on target parallel region.
  //
  ZERO(A);
  {
    double p[1], q = 99;
    p[0] = 1;
    A[0] = p[0];
    TESTD("omp target parallel private(p, q) num_threads(1)", {
        p[0] = 2;
        q = 0;
        A[0] += p[0];
    }, VERIFY(0, 1, A[i]+p[0]+q, (1+trial)*2+2+99));
  }

  //
  // Test: firstprivate clause on parallel region.
  //
  ZERO(A);
  {
    double p[1], q = 99;
    p[0] = 5;
    A[0] = p[0];
    TESTD("omp target parallel firstprivate(p, q) num_threads(1)", {
        A[0] += p[0] + q;
        p[0] = 2;
        q = 0;
    }, VERIFY(0, 1, A[i]+p[0]+q, (1+trial)*(99+5)+5+5+99));
  }

#if 0
  INCORRECT CODEGEN
  //
  // Test: shared clause on parallel region.
  //
  ZERO(A);
  {
    double p[1], q;
    p[0] = 5;
    A[0] = p[0];
    q = -7;
    TESTD("omp target parallel num_threads(2) shared(p, q)", {
        if (omp_get_thread_num() == 1) {
          p[0] = 99; q = 2;
        }
        _Pragma("omp barrier")
        if (omp_get_thread_num() == 0)
          A[0] += p[0] + q;
        _Pragma("omp barrier")
        p[0] = 1; q = -100;
    }, VERIFY(0, 1, A[i]+p[0]+q, (1+trial)*(99+2)+5+-7));
  }
#endif

  return 0;
}
