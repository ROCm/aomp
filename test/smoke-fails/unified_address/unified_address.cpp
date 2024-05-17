#include <cstdio>

#pragma omp requires unified_address

int main() {
  int n = 1000;
  int *a = new int[n];

  // init
  for(int i = 0; i < n; i++)
    a[i] = i;

  #pragma omp target teams distribute parallel for map(a[:n])
  for(int i = 0; i < n; i++)
    a[i] += 1;

  //check
  int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != i+1) {
      err++;
      printf("Err at %d, got %d expected %d\n", i, a[i], i+1);
      if(err > 10)
	return err;
    }
  return err;
}
