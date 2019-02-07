
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

#define PARALLEL() {                        \
  _Pragma("omp parallel num_threads(128)")  \
    {                                       \
      int i = omp_get_thread_num()*4;       \
      for (int j = i; j < i + 4; j++) {     \
        A[j] += C[j] + D[j];                \
      }                                     \
    }                                       \
}

// Not sure how to invoke a macro multiple times
#define PARALLEL5() { PARALLEL() PARALLEL() PARALLEL() PARALLEL() PARALLEL() }
#define PARALLEL25() { PARALLEL5() PARALLEL5() PARALLEL5() PARALLEL5() PARALLEL5() }
#define PARALLEL125() { PARALLEL25() PARALLEL25() PARALLEL25() PARALLEL25() PARALLEL25() }

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];

  INIT();

  //
  // Test: Multiple parallel regions in a single target.
  //
  TEST({
   for (int i = 0; i < 512; i++) {
     A[i] = 0;
   }
   PARALLEL125()
  }, VERIFY(0, 512, A[i], 125*(1+i)));

  return 0;
}
