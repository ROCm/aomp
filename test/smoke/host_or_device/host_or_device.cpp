#include <cstdio>
#include <cstdlib>

#define LARGE 1000000
#define SMALL 100
#define THRESHOLD 10000

void init(size_t n, double *a, double *b, double *c) {
  for(size_t i = 0; i < n; i++) {
    a[i] = 0.0;
    b[i] = (double) i;
    c[i] = (double) i+3;
  }
}
int main() {
  // n may be a large or small number
  size_t n = (rand() > (RAND_MAX/2)) ? LARGE : SMALL;

  double *a = new double[n];
  double *b = new double[n];
  double *c = new double[n];
  double k = 3.14;

  printf("n = %zu\n", n);

  init(n, a, b, c);

  // decide if we want host or device execution based on value of n
  #pragma omp target teams distribute parallel for simd if(target: n>THRESHOLD) \
    map(from:a[:n]) map(to: b[:n], c[:n])
  for(int i = 0; i < n; i++)
    a[i] = b[i] + k*c[i];

  printf("%lf\n", a[0]);

  delete [] a;
  delete [] b;
  delete [] c;

  if (n == LARGE) n = SMALL;
  else n = LARGE;

  printf("n = %zu\n", n);

  a = new double[n];
  b = new double[n];
  c = new double[n];

  init(n, a, b, c);

  // decide if we want host or device execution based on value of n
  #pragma omp target teams distribute parallel for simd if(target: n>THRESHOLD) \
    map(from:a[:n]) map(to: b[:n], c[:n])
  for(int i = 0; i < n; i++)
    a[i] = b[i] - k*c[i];

  printf("%lf\n", a[0]);

  delete [] a;
  delete [] b;
  delete [] c;

  return 0;
}
