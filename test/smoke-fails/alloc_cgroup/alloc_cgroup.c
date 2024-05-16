//===--- test_target_uses_allocators_cgroup.c -----------------------------===//
//
// OpenMP API Version 5.0 Nov 2018
//
// The tests checks the uses_allocators clause with omp_cgroup_mem_alloc. 
// The variable allaocated in the target is modified and used to compute result on 
// device. Result is copied back to the host and checked with computed value on host.
//
//===----------------------------------------------------------------------===//

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#define N 1024

int test_uses_allocators_cgroup() {
  int errors = 0;
  int x = 0;
  int device_result = 0;
  int result = 0;

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      result += j + i ;
    }
  }

#pragma omp target uses_allocators(omp_cgroup_mem_alloc) allocate(omp_cgroup_mem_alloc: x) firstprivate(x) map(from: device_result)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      x += j + i;
    }
  }
  device_result = x;
}
return result != device_result;

}

int main() {

  int errors = test_uses_allocators_cgroup();
  if (errors) printf("Failed\n");
  return errors;
}

