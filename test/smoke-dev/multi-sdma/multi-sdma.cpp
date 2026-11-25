#include <cstdio>

#define N 1024
// Enough copies to go around the engine set several times, even if ROCr
// offers many engines for one direction.
#define NUM_RR_COPIES 64

int main(int argc, char **argv) {
  int Errors = 0;

  int *pA = new int[N];
  int *pB = new int[N];
  double *pD = new double[N];

  // Set-up: loads the image, does all init-time transfers and fills the device
  // copy of pA before the round-robin window starts.
#pragma omp target enter data map(alloc: pA[0:N])
#pragma omp target
  for (int i = 0; i < N; ++i)
    pA[i] = i;

  // Round-robin window: only device-to-host copies, strictly one after
  // another. This keeps the engine mask reported by
  // hsa_amd_memory_copy_engine_status fixed, so the plugin has to cycle
  // through that (unknown) set of engines in a fixed order.
  // round-robin.awk checks the engine sequence between the two markers. Debug
  // output goes to unbuffered stderr, so print the markers there too.
  fprintf(stderr, "multi-sdma: begin round-robin\n");
  for (int c = 0; c < NUM_RR_COPIES; ++c) {
    for (int i = 0; i < N; ++i)
      pA[i] = -1;
#pragma omp target update from(pA[0:N])
    for (int i = 0; i < N; ++i)
      Errors += pA[i] != i;
  }
  fprintf(stderr, "multi-sdma: end round-robin\n");
#pragma omp target exit data map(delete: pA[0:N])

  // Mixed directions: only checks that the results are correct.
#pragma omp target teams distribute parallel for map(from: pA[0:N])
  for (int i = 0; i < N; ++i)
    pA[i] = 1;

#pragma omp target teams distribute parallel for map(from: pB[0:N])
  for (int i = 0; i < N; ++i)
    pB[i] = 2;

#pragma omp target teams distribute parallel for map(tofrom: pA[0:N], pB[0:N]) map(from: pD[0:N])
  for (int i = 0; i < N; ++i) {
    pD[i] = pA[i] * pB[i];
  }

  for (int i = 0; i < N; ++i)
    Errors += pD[i] != 2;

  fprintf(stderr, "multi-sdma: %s\n", Errors ? "FAIL" : "PASS");
  return Errors ? 7 : 0;
}

// CHECK: multi-sdma: PASS
// CHECK: round-robin: OK
