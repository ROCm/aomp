#include <cstdio>
#include<omp.h>

#pragma omp requires unified_shared_memory

#define N 10000

int main() {
  int *a = new int[N]; // mapped
  int *b = new int[N]; // not mapped

  // implicitly mapping to device(0)
  #pragma omp target enter data map(to:a[:N])

  // should be mapped
  if(!omp_get_mapped_ptr(a, /*device_num=*/0)) return 1;

  // wrong device
  if(omp_get_mapped_ptr(a, /*device_num=*/1)) return 1;

  // not mapped
  if(omp_get_mapped_ptr(b, /*device_num=*/0)) return 1;
  printf("a = %p\n", a);

  // a is a on host, but libomptarget it still thinks it is an error to use omp_get_initial_device
  // even though returning a is required by the spec's
  if((omp_get_mapped_ptr(a, /*device_num=*/omp_get_initial_device()) != a)) return 1;
  
  
  return 0;
}
