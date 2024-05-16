#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1000

int test_all_dependencies() {

  int errors = 0;
  int dep_1[N], dep_2[N];

  // Initialize dep_1 and dep_2
  for (int i = 0; i < N; ++i) {
    dep_1[i] = 0;
    dep_2[i] = 0;
  }

#pragma omp target depend(out: dep_1) map(tofrom: dep_1[0:N])
  {
    for (int i = 0; i < N; i++) {
      dep_1[i] = 1;
    }
  } // end of omp target

  #pragma omp taskwait

  return 0;
}

int main() {
  int errors = 0;
  test_all_dependencies();
  return 0;
}

