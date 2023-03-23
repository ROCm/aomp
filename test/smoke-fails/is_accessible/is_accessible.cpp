#include<cstdio>
#include<omp.h>

int main() {
  int n = 10000;
  int *a = new int[n];
  // program must be executed with HSA_XNACK=1
  if (!omp_target_is_accessible(a, n*sizeof(int), /*device_num=*/0)) return 1;
  #pragma omp target teams distribute parallel for
  for(int i = 0; i < n; i++)
    a[i] = i;

  // check
    int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != i) {
      err++;
      printf("Error at %d: got %d expected %d\n", i, a[i], i);
      if (err > 10) return err;
    }
  return err;
}
