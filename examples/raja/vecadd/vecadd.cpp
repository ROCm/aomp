#include <stdio.h>
#include <omp.h>

#include "RAJA/RAJA.hpp"

using namespace RAJA;
using namespace RAJA::statement;

#define N 100
int main()
{

  int* a = new int[N*N];
  int* b = new int[N*N];
  int* r = new int[N*N];

  int i;

  for (i=0; i<N*N; i++)
    a[i]=i;

  for (i=0; i<N*N; i++)
    b[i]=i*2;

  for(int i = 0; i < N; ++i) {
    for(int j = 0; j < N; ++j) {
      r[i + (N*j)] = a[i + (N*j)] + b[i + (N*j)];
    }
  }

   using Pol = KernelPolicy<
    Collapse<omp_target_parallel_collapse_exec, ArgList<0,1>, Lambda<0> > >;

#pragma omp target data map(tofrom: a[0:N*N]) map(to: b[0:N*N]) use_device_ptr(a) use_device_ptr(b)
  RAJA::kernel<Pol>(
      RAJA::make_tuple(
        RAJA::RangeSegment(0,N),
        RAJA::RangeSegment(0,N)),
      [=] (int i, int j) {
	a[i + (N*j)] = a[i + (N*j)] + b[i + (N*j)];
      });

  // Check Result
  int rc = 0;
  for(int w = 0; w < N*N; ++w) {
    if (a[w] != r[w]) {
      printf("Error a[%d] = %d, expected %d\n",w, a[w], r[w]);
    }
  }

  if (!rc)
    printf("Success\n");

  return rc;
}

