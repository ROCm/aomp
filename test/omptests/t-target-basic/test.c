
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

int main(void){
  check_offloading();

  int fail;
  double A[N], B[N], C[N], D[N], E[N];

  INIT();

#if 0
  //
  // Test: Execute on host
  //
  #pragma omp target if (target: C[0] == 0)
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      A[i] = C[i] + D[i] + omp_is_initial_device();
  }

  fail = 0;
  VERIFY(0, N, A[i], i+2);
  if (fail) {
    printf ("Test1: Failed\n");
  } else {
    printf ("Test1: Succeeded\n");
  }
#endif

  //
  // Test: Execute on device
  //
  #pragma omp target device(1) if (target: C[0] == 1)
  {
    #pragma omp parallel for schedule(static,1)
    for (int i = 0; i < 992; i++)
      A[i] = C[i] + D[i] + /*omp_is_initial_device()=*/1;
    // We cannot use omp_is_initial_device() directly because this is tested for
    // the host too.
  }

  // CHECK: Succeeded
  fail = 0;
  VERIFY(0, N, A[i], i+2);
  if (fail) {
    printf ("Test2: Failed\n");
  } else {
    printf ("Test2: Succeeded\n");
  }

  //
  // Test: Printf on device
  //
  #pragma omp target
  {
    printf ("Master %d\n", omp_get_thread_num());
    int TT[2] = {0,0};
    #pragma omp parallel num_threads(2)
    {
      TT[omp_get_thread_num()]++;
    }
    printf ("Parallel %d:%f\n", TT[0], D[0]);
    printf ("Parallel %d:%f\n", TT[1], D[1]);
  }

  return 0;
}
