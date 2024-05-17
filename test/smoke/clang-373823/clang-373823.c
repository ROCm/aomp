#include <stdio.h>

// Check that function called from target region is set to "rpc:0"

#define N 100
void computeFoo(double *Ptr)
{
  int stack[N];
  for(int i = 0; i < N;  i++) {
    stack[i] = i;
    // required to force rpc:1
    Ptr[i] += 1;
  }
  // required to force rpc:1
  Ptr[0] = stack[N-1] + 1.0;
}

int computeFindNeighbors()
{
  double a[N] = {0.0};
  int j;

  #pragma omp target teams distribute parallel for
  for (j = 0; j< N; j++) {
    computeFoo(a);
  }

  int fail = 0;
  if (a[0] != 100.0) {
    fail++;
    printf ("Wrong value: a[%d]=%lf\n", 0, a[0]);
  }

  if (!fail)
    printf("Success\n");
  return fail;
}

int main() {
  return computeFindNeighbors();
}

/// CHECK: rpc:0
