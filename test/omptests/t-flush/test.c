
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

  INIT();

  //
  // Test: Flush.
  // FIXME: Hard to test multiple threads on the GPU with this example.
  // With a single thread, variable 'done' should be constantly read from memory
  // in the while loop.
  //
  int done[1];
  TEST({
    int done = 0;
    for (int i = 0; i < N; i++) {
      B[i] = -1;
    }
    _Pragma("omp parallel if(0)")
    {
      _Pragma("omp sections")
      {
        _Pragma("omp section")
        {
          for (int i = 0; i < N; i++) {
            B[i] = C[i] + D[i];
          }
          _Pragma("omp flush(B)")
          done = 1;
          _Pragma("omp flush(done)")
        }
        _Pragma("omp section")
        {
          while (!done) {
            _Pragma("omp flush(done)")
          }
          _Pragma("omp flush(B)")
          for (int i = 0; i < N; i++) {
            A[i] = B[i] + D[i] + E[i];
          }
        }
      }
    }
  }, VERIFY(0, N, A[i], i + 1));

  return 0;
}
