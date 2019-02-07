#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 1; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];
  double *ptrA = &A[0];
  unsigned long long addr1;
  unsigned long long addr2;

  INIT();

  //
  // Test: Master.
  //
  for (int t = 0; t < TRIALS; t++) {
    int threads[1];
    threads[0] = t;

    // init
    for (int i = 0; i < N; i++) {
      A[i] = 1.0;
    }

    // compute
    #pragma omp target data map(A)
    {
      #pragma omp target data map(B) use_device_ptr(ptrA)
      {
        addr1 = (unsigned long long)((void *) ptrA);
        #pragma omp target is_device_ptr(ptrA) map(to: D, E) map(from: addr2)
        {
          addr2 = (unsigned long long)((void *) ptrA);
          for (int i = 0; i < N; i++) {
            ptrA[i] += D[i] - E[i];
          }
        }
      }
    }

    int error = 0;
    if (addr1 != addr2)
      printf("Address of A: 0x%llx, ptrA on host 0x%llx, in use device 0x%llx, "
          "on device 0x%llx, error %d\n", (unsigned long long)((void *)&A[0]),
          (unsigned long long)((void *) ptrA), addr1, addr2, ++error);

    for (int i = 0; i < N; i++) {
      if (A[i] != 2.0*i+1.0)
        printf("%d: got %f, wanted %f, error %d\n", i, A[i], 2.0*i+1.0,
            ++error);
    }

    if (error)
      printf("Failed with %d errors\n", error);
    else
      printf("Success\n");
  }

  return 0;
}
