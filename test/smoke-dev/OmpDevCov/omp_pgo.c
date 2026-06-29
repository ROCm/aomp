/* OmpDevCov: device-side PGO round trip for OpenMP target offload.
 *
 * OpenMP offload has no HIP-style host-shadow drain for its device code, so the
 * only way device counters reach the .profraw is the in-tree HSA-introspection
 * drain. A successful -fprofile-generate -> llvm-profdata merge -> -fprofile-use
 * cycle that consumes a device function's counters therefore exercises the HSA
 * drain exclusively.
 *
 * classify() runs on the device and has a data-dependent branch so there is a
 * non-trivial counter to instrument, merge, and consume.
 */

#include <stdio.h>
#include <stdlib.h>

#define N 4096

#pragma omp declare target
static int classify(int x) {
  if ((x & 1) == 0)
    return x * 2; /* even path */
  else
    return x + 1; /* odd path  */
}
#pragma omp end declare target

int main(void) {
  int *a = (int *)malloc(N * sizeof(int));
  if (!a)
    return 1;
  for (int i = 0; i < N; i++)
    a[i] = i;

#pragma omp target teams distribute parallel for map(tofrom : a[0:N])
  for (int i = 0; i < N; i++)
    a[i] = classify(a[i]);

  long sum = 0;
  for (int i = 0; i < N; i++)
    sum += a[i];

  printf("sum=%ld\n", sum);
  free(a);
  return 0;
}
