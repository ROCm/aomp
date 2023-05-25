#include<cstdio>

#define N 1024

void foo(int *a, int n) {
  #pragma omp target teams distribute parallel for has_device_addr(a)
  for(int i = 0; i < n; i++)
    a[i] = i;
}

int main() {
  int *a = new int[N];

  #pragma omp target enter data map(alloc:a[:N])
  foo(a, N);
  #pragma omp target enter data map(release:a)

  // check
  int err = 0;
  for(int i = 0; i < N; i++)
    if (a[i] != i) {
      err++;
      printf("Error at %d: got %d expected %d\n", i, a[i], i);
      if (err > 10) return err;
    }
  return err;

}
