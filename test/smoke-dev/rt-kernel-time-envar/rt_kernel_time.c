#include <omp.h>
#include <stdio.h>

int main() {
  int N = 10;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

#pragma omp parallel for num_threads(4)
  for (int z = 0; z < 10; z++)
#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// This test checks whether unique launch ids will be generated for concurrent
/// kernel launch when LIBOMPTARGET_KERNEL_EXE_TIME is enabled.
/// CHECK: 10
