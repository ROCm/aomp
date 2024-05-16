#include <omp.h>
#include <stdio.h>

#define THREADS 512
#ifdef WAVE_SIZE 
  #define WARPSIZE WAVE_SIZE
#else
  #define WARPSIZE 64
#endif

#pragma omp declare target
omp_lock_t lock;
#pragma omp end declare target

int main() {

  if (WARPSIZE == 32)
    return 0;

  int error = 0;
  unsigned count = 0;          // incremented within target region
  unsigned expected_count = 0; // incremented on host

#pragma omp target
  omp_init_lock(&lock);


  // The lock implementation picks a thread from the warp to avoid the
  // deadlock that results if multiple threads try to CAS-loop at once

  // The lower/upper construct checks various active warp patterns

  const int edges[] = {0, 1, 32, 62, 63};
  const int N = sizeof(edges) / sizeof(edges[0]);
  for (int l = 0; l < N; l++) {
    for (int u = 0; u < N; u++) {
      int lower = edges[l];
      int upper = edges[u];
      if (lower > upper)
        continue;

      expected_count += THREADS / WARPSIZE;

#pragma omp target parallel num_threads(THREADS) map(tofrom : error, count)
      {
        int lane_id = omp_ext_get_lane_id();
        if (lane_id >= lower && lane_id <= upper) {

          omp_set_lock(&lock); // mutex acts on a per warp basis

          if (omp_ext_get_lane_id() == lower) {
            // Increment once per warp
            count++;
          }

          if (!omp_test_lock(&lock)) {
            error = 1;
          }

          omp_unset_lock(&lock);
        }
      }
    }
  }

#pragma omp target
  omp_destroy_lock(&lock);

  if (count != expected_count) {
    error = 1;
  }

  fprintf(stderr, "ec %d c %d\n", expected_count, count);
  return error;
}
