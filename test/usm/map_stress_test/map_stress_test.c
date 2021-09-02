#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <omp.h>

#define N 100993

// to run this correctly, use
// export HSA_XNACK=1, which implies
// pragma omp requires unified_shared_memory

int main() {
  int n = N;

  int *a = (int *)malloc(n *sizeof(int));
  double *b = (double *) omp_target_alloc(n*sizeof(double), omp_get_default_device());
  float *c = (float *)malloc(n *sizeof(float));

  #pragma omp target teams distribute parallel for map(from: a[:n]) map(close, from: c[:n])
  for(int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i+1;
    c[i] = i+2;
  }

  int err = 0;
  for(int i = 0; i < n; i++) {
    if(a[i] != i || b[i] != (i+1) || c[i] != (i+2)) {
      printf("%d: a = %d (%d), b = %lf (%lf), c = %f (%f)\n", i, a[i], i, b[i], (double) i+1, c[i], (float) i+2);
      err++;
      if(err > 10) break;
    }
  }

  #pragma omp target map(a[:n]) map(close, tofrom: c[:n])
  for(int i = 0; i < n; i++) {
    a[i] += i;
    b[i] += i+1;
    c[i] += i+2;
  }

  // check
  err = 0;
  for(int i = 0; i < n; i++) {
    if(a[i] != 2*i || b[i] != (2*i+2) || c[i] != (2*i+4)) {
      printf("%d: a = %d (%d), b = %lf (%lf), c = %f (%f)\n", i, a[i], 2*i, b[i], (double) 2*i+2, c[i], (float) 2*i+4);
      err++;
      if(err > 10) break;
    }
  }

  if(err) printf("There were errors\n");

  omp_target_free(b, omp_get_default_device());

  return err;
}
