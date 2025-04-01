#include<cstdio>

#define N 1024

#pragma omp requires unified_shared_memory

int main() {
  double *a = new double[N*N];

  #pragma omp parallel for collapse(2)
  for(int i = 0; i < N; i++)
    for(int j = 0; j < N; j++)
      a[i*N+j] = (double) (i*N+j);

  #pragma omp target teams distribute parallel for collapse(2)
  for(int i = 0; i < N; i++) {
    double k = i*3.14;
    for(int j = 0; j < N; j++)
      a[i*N+j] += k;
  }

  //check
  int err = 0;
  for(int i = 0; i < N; i++) {
    double k = i*3.14;
    for(int j = 0; j < N; j++)
      if (a[i*N+j] != (double) (i*N+j) + k) {
        err++;
        printf("Error at (%d,%d): got %lf expected %lf\n", i, j, a[i*N+j], (double) (i*N+j) + k);
        if (err > 10) return err;
      }
  }

  return err;
}
