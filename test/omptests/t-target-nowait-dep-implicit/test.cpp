#include <stdlib.h>
#include <stdio.h>

#define N 1000

int main() {
  double *a, *ah, *b, *c, k, l;
  double z[N];
  int n = N;

  a = (double *) malloc(N*sizeof(double));
  ah = (double *) malloc(N*sizeof(double));
  b = (double *) malloc(N*sizeof(double));
  c = (double *) malloc(N*sizeof(double));

  k = 2.7803;
  l = 82.348;

  for(int i = 0; i < N; i++) {
    a[i] = ah[i] = i;
    b[i] = i*2;
    c[i] = i-10;
    z[i] = (i-10)*2;
  }
  
  #pragma omp target enter data map(to:a[:n], b[:n], c[:n]) depend(out: a, b, c)

  #pragma omp target depend(in:b, c) depend(inout: a) firstprivate(l)
  {
    #pragma omp teams distribute parallel for
    for(int i = 0; i < N; i++)
      a[i] += b[i] + k*c[i] + l*z[i];
  }
  
  for(int i = 0; i < N; i++) {
    ah[i] += b[i] + k*c[i] + l*z[i];
  }

  bool err = false;
  #pragma omp target exit data map(from:a[:n]) depend(in: a)
  for(int i = 0; i < N; i++) {
    if (a[i] != ah[i]) {
      err = true;
      printf("Error at %d, host = %lf, device = %lf\n", i, ah[i], a[i]);
    }
  }

  if (err) printf("Errors!\n");
  else printf("Success!\n");

  return 0;
}
