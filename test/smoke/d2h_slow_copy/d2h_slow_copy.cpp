#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>

#define N 30000

using namespace std;

struct timer {
  const char *func;
  using clock_ty = std::chrono::high_resolution_clock;
  std::chrono::time_point<clock_ty> start, next;

  explicit timer(const char *func): func(func) {
    start = clock_ty::now();
    next = start;
  }

  void checkpoint(const char *func) {
    auto end = clock_ty::now();

    uint64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - next)
            .count();

    printf("%35s: %lu ns (%f ms)\n", func, t, t / 1e6);
    next = clock_ty::now();
  }

  ~timer() {
    auto end = clock_ty::now();

    uint64_t t =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count();

    printf("%35s: %lu ns (%f ms)\n", func, t, t / 1e6);
  }
};

int main() {
  float *arr = (float*)malloc(sizeof(float) * N * N);
  timer stopwatch("main()");

  #pragma omp target
  {
    printf("Hello, world\n");
  }

  stopwatch.checkpoint("First kernel");
  #pragma omp target data map(tofrom:arr[0:N*N])
  {
    stopwatch.checkpoint("Host to device copy");
    #pragma omp target teams
    #pragma omp distribute
    for (int i = 0; i < N; i++) {
      #pragma omp parallel for
      for (int j = 0; j < N; j++) {
        arr[i*N + j] = sqrt((i * j) & 0xfffff);
      }
    }

    stopwatch.checkpoint("Compute");
  }

  stopwatch.checkpoint("Device to host");
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      float expected = (i * j) & 0xfffff;
      float val = arr[i *N + j];
      if (round(val * val) != (float)expected) {
        printf("ERROR at (%d, %d) = %f != %f\n", i, j, val * val, (float)(expected));
        exit(1);
      }
    }

  stopwatch.checkpoint("Results verified");
  printf("Done\n");
  free(arr);
  return 0;
}
