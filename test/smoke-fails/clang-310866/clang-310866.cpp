#include <omp.h>
#include <stdio.h>
// lifted from sollve 5.0 concurrent test
int main(){
#define N 1024

  int E[N];
  int x[N];
  int y[N];
  int z[N];
  int num_threads = -1;

  for (int i = 0; i < N; i++) {
    x[i] = 1;
    E[i] = x[i];
    y[i] = i + 1;
    z[i] = 2*(i + 1);
  }
  for (int i = 0; i < N; i++) {
    E[i] += y[i]*z[i];
  }
#pragma omp parallel num_threads(8)
  {
#pragma omp loop order(concurrent)
    for (int i = 0; i < N; i++) {
      x[i] += y[i]*z[i];
    }
  }
  for (int i = 0; i < N; i++) {
    if (E[i] != x[i]) return 1;
  }
  return 0;
}
