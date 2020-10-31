#include <omp.h>
#include <stdio.h>
#include <chrono>
#include <memory>

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
  int *arr = new int[N*N];
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
        arr[i*N + j] = i + j;
      }
    }

    stopwatch.checkpoint("Compute");
  }

  stopwatch.checkpoint("Device to host");
  for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++) {
      if (arr[i*N + j] != i+j) {
        fprintf(stderr, "arr[%d*%d + %d] != %d+%d (%d != %d)\n", i, N, j, i, j, arr[i*N+j], i+j);
        delete[] arr;
        return 1;
      }
    }

  stopwatch.checkpoint("Results verified");
  printf("Done\n");
  delete[] arr;
  return 0;
}
