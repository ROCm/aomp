#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define N 1024

int testTargetTeamsLoopIf(int isTrue) {
  int a[N], b[N], c[N];
  int errors = 0;
  // Data Inititalize
  for (int i = 0; i < N; i++) {
    a[i] = 2*i;  // Even
    b[i] = 2*i + 1;  // Odd
    c[i] = 0;
  }
  // Execute on target
#pragma omp target teams loop map(to: a[0:N], b[0:N]) map(from: c[0:N]) if(isTrue)
  for (int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }

  for (int i = 0; i < N; i++) {
    assert(c[i] == (a[i] + b[i]));
  }
  return errors;
}

int main() {
  int errors = 0;
  errors = testTargetTeamsLoopIf(0);
  errors = testTargetTeamsLoopIf(1);
  return errors;
}
