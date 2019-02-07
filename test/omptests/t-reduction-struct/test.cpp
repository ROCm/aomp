
#include <stdio.h>
#include <float.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (957*3)

#define ZERO(X) ZERO_ARRAY(N, X) 

#pragma omp declare target
struct BaseTy {
  double A;
  double B;
  BaseTy() : A(0), B(0) {}
};

struct DataTy : BaseTy {
  int C;
  int D;
  double E;
  DataTy() : C(0), D(0), E(0) {}
};

struct DataTy merge_data( struct DataTy omp_in, struct DataTy omp_out) {
  omp_out.A += omp_in.A;
  omp_out.B += omp_in.B;
  omp_out.C += omp_in.C;
  omp_out.D += omp_in.D;
  omp_out.E += omp_in.E;
  return omp_out;
}
#pragma omp end declare target

#pragma omp declare reduction (merge: DataTy : omp_out = merge_data(omp_in, omp_out))


#define INIT1 (1)
#define INIT2 (3)
#define INIT3 (5)
#define INIT4 (7)
#define INIT5 (9)

#define EXPECTED_RESULT ( \
INIT1 + INIT2 + INIT3 + \
INIT4 + INIT5 + \
N*5 \
)

#define REDUCTION_CLAUSES reduction(merge: Data)

#define REDUCTION_MAP map(tofrom: Data)

#define REDUCTION_INIT() {           \
      Data.A = INIT1; Data.B = INIT2;      \
      Data.C = INIT3; Data.D = INIT4;      \
      Data.E = INIT5;    \
}

#define REDUCTION_BODY() \
      Data.A += 1; Data.B += 1; \
      Data.C += 1; Data.D += 1; \
      Data.E += 1;


#define REDUCTION_LOOP() \
    for (int i = 0; i < N; i++) { \
      REDUCTION_BODY(); \
    }

#define REDUCTION_FINAL() { \
      OUT[0] += (long long) (Data.A + Data.B + Data.C + Data.D + Data.E); \
}

int main(void) {
  check_offloading();

  struct DataTy Data;
  long long OUT[1];
  long long EXPECTED[1];
  EXPECTED[0] = EXPECTED_RESULT;

  int cpuExec = 0;
  #pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }
  int gpu_threads = 512;
  int cpu_threads = 32;
  int max_threads = cpuExec ? cpu_threads : gpu_threads;

  //
  // Test: reduction on target teams distribute parallel for.
  //
  for (int tms = 1 ; tms <= 512 ; tms *= 7) {
    for (int ths = 1 ; ths <= 1024 ; ths *= 9) {
      OUT[0] = 0;
      TESTD2("omp target teams distribute parallel for num_teams(tms) thread_limit(ths) REDUCTION_MAP REDUCTION_CLAUSES",
        {
          REDUCTION_INIT();
        },
        REDUCTION_LOOP(),
        {
          REDUCTION_FINAL();
        },
        VERIFY(0, 1, OUT[i], (trial+1) * EXPECTED[i]));
    }
  }

  return 0;
}
