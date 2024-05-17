#include <stdio.h>
int main() {
  int N = 100000;
  double a[N];
  double my_min = (double) N;
  for (int i=0; i<N; i++)
    a[i] = (double) i + 1;

#pragma omp target teams distribute parallel for map(tofrom:my_min) map(to:a) reduction(min:my_min)
  for (int j = 0; j< N; j++)
    my_min = (a[j]<my_min) ? a[j] : my_min;
  
  int rc = (my_min != 1.0) ;
  if (rc)
    printf("FAIL!  expected:1.0  OpenMP min: %f\n", my_min);
  else
    printf("Success\n");
  return rc;
}
