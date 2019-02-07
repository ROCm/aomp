
#include <stdio.h>
#include <omp.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define TRIALS (1)

#define N (992)

#define INIT() INIT_LOOP(N, {C[i] = 0; D[i] = i; E[i] = -i;})

#define ZERO(X) ZERO_ARRAY(N, X) 

#define SECTION(i) \
  _Pragma("omp section") \
  A[i] = C[i] + D[i];

#define SECTION5(i) SECTION(i) SECTION(i+1) SECTION(i+2) SECTION(i+3) SECTION(i+4)
#define SECTION25(i) SECTION5(i) SECTION5(i+5) SECTION5(i+10) SECTION5(i+15) SECTION5(i+20)
#define SECTION125(i) SECTION25(i) SECTION25(i+25) SECTION25(i+50) SECTION25(i+75) SECTION25(i+100)

int main(void) {
  check_offloading();

  double A[N], B[N], C[N], D[N], E[N];

  INIT();

  //
  // Test: Sections in a parallel region with two section regions.
  //
  TEST({
    _Pragma("omp parallel num_threads(13)")
    {
      _Pragma("omp sections")
      {
      _Pragma("omp section")
      for (int i = 0; i < N; i++)
        A[i] = C[i] + D[i];
      _Pragma("omp section")
      for (int i = 0; i < N; i++)
        B[i] = D[i] + E[i];
      }
    }
  }, VERIFY(0, N, B[i], A[i] - i));

  //
  // Test: Sections in a serialized parallel region with two section regions.
  //
  TEST({
    _Pragma("omp parallel num_threads(13) if(0)")
    {
      _Pragma("omp sections")
      {
      _Pragma("omp section")
      for (int i = 0; i < N; i++)
        A[i] = C[i] + D[i];
      _Pragma("omp section")
      for (int i = 0; i < N; i++)
        B[i] = D[i] + E[i];
      }
    }
  }, VERIFY(0, N, B[i], A[i] - i));

  //
  // Test: Large number of section regions.
  //
  for (int t = 0; t <= 224; t++) {
    int threads[1]; threads[0] = t;
    TEST({
      _Pragma("omp parallel num_threads(threads[0]) if(parallel: threads[0] > 1)")
      {
        _Pragma("omp sections")
        {
          SECTION125(0)
        }
      }
    }, VERIFY(0, 125, A[i], i));
  }

  //
  // Test: Private clause on Sections.
  //
  TEST({
    int i;
    _Pragma("omp parallel num_threads(13)")
    {
      _Pragma("omp sections private(B, i)")
      {
      _Pragma("omp section")
      for (i = 0; i < N; i++)
        A[i] = C[i] + D[i];
      _Pragma("omp section")
      for (i = 0; i < N; i++)
        B[i] = D[i] + E[i];
      }
    }
    for (i = 0; i < N; i++) {
      B[i] = -1;
    }
  }, VERIFY(0, N, B[i]+1, A[i] - i));

  //
  // Test: First private clause on Sections.
  //
  TEST({
    int i;
    for (i = 0; i < N; i++) {
      B[i] = -1;
    }
    _Pragma("omp parallel num_threads(13)")
    {
      _Pragma("omp sections firstprivate(B) private(i)")
      {
      _Pragma("omp section")
      for (i = 0; i < N; i++)
        A[i] = B[i] + C[i] + D[i];
      _Pragma("omp section")
      for (i = 0; i < N; i++)
        B[i] += D[i] + E[i];
      }
    }
  }, VERIFY(0, N, B[i], A[i] - i));

  //
  // Test: Last private clause on Sections.
  //
  TEST({
    int i;
    for (i = 0; i < N; i++) {
      B[i] = -1;
    }
    _Pragma("omp parallel num_threads(13)")
    {
      _Pragma("omp sections firstprivate(B) lastprivate(B) private(i)")
      {
      _Pragma("omp section")
      for (i = 0; i < N; i++) {
        B[i] = C[i] + 1;
        A[i] = B[i] + D[i];
      }
      _Pragma("omp section")
      for (i = 0; i < N; i++)
        B[i] += D[i] + E[i];
      }
    }
  }, VERIFY(0, N, B[i] + 1, A[i] - i - 1));

  //
  // Test: Requirement of a barrier after a sections region.
  //
  TEST({
    int i;
    _Pragma("omp parallel num_threads(224)")
    {
      _Pragma("omp sections private(i)")
      {
        SECTION125(0)
      }
      if (omp_get_thread_num() == omp_get_num_threads()-1) {
        for (i = 0; i < 125; i++) {
          B[i] = A[i] + D[i] + E[i];
        }
      }
    }
  }, VERIFY(0, 125, B[i], i));

  //
  // Test: Nowait in a sections region.
  // FIXME: Not sure how to test for correctness.
  //
  TEST({
    int i;
    _Pragma("omp parallel num_threads(224)")
    {
      _Pragma("omp sections nowait private(i)")
      {
        SECTION125(0)
      }
      _Pragma("omp for nowait schedule(static,1)")
      for (i = 0; i < 125; i++) {
        B[i] = A[i] + D[i] + E[i];
      }
    }
  }, VERIFY(0, 125, B[i], i));

}
