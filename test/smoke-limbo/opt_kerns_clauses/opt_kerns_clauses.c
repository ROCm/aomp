#include <stdio.h>
#include <omp.h>

int main()
{
  int use_device = 1;
  int N = 100000;
  double a[N], b[N];
  double sum1 = 0;
  int scalar1 = 10;

  for (int i=0; i<N; i++) 
    a[i]=i;

  
#pragma omp target teams distribute parallel for \
  device(0), if(use_device) order(concurrent) \
  defaultmap(firstprivate:scalar) defaultmap(to:aggregate) \
  default(none) firstprivate(N,scalar1,use_device), shared(b,a) \
  map(from:b)
  for (int k = 0; k< N; k++)
    b[k]=a[k] * scalar1;
  
#pragma omp target teams distribute parallel for \
  device(0), if(use_device) order(concurrent) \
  defaultmap(firstprivate:scalar) defaultmap(to:aggregate) \
  default(none) firstprivate(N,scalar1,use_device), shared(a) \
  map(tofrom:sum1) reduction(+:sum1)
  for (int j = 0; j< N; j=j+1)
    sum1 += a[j] + scalar1;

  printf("%f\n", sum1);

  int rc = sum1 != 5000950000;
  for (int i=0; i<N; i++)
    if (b[i] != a[i] * scalar1 ) {
      rc++;
      printf ("Wrong value: b[%d]=%f\n", i, b[i]);
    }
  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:4
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8

