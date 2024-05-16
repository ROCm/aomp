#include<cstdio>
#include<omp.h>

#pragma omp requires dynamic_allocators

int main() {
  int n = 1024;
  int r = 0;
  #pragma omp target map(tofrom:r)
  {
    omp_allocator_handle_t al = omp_init_allocator(omp_default_mem_space, 0, {});

    int *p = (int *)omp_alloc(n*sizeof(int), al);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
      p[i] = i;

    #pragma omp parallel for reduction(+:r)
    for(int i = 0; i < n; i++)
      r += p[i];

    omp_free(p, al);
  }

  // sum of first n numbers starting from 0 is is (n-1)*n/2
  int res = (n-1)*n/2;
  if (r != res) {
    printf("Error: reduction value is %d and should be %d\n", r, res);
    return 1;
  }

  int *q = nullptr;
  #pragma omp target map(from:q)
  {
    omp_allocator_handle_t al = omp_init_allocator(omp_default_mem_space, 0, {});

    int *p = (int *)omp_alloc(n*sizeof(int), al);
    #pragma omp parallel for
    for(int i = 0; i < n; i++)
      p[i] = i;
    q = p;
  }
  if (!q) {
    printf("q was not assigned device pointer p\n");
    return 1;
  }

  #pragma omp target parallel for is_device_ptr(q)
  for(int i = 0; i < n; i++)
    q[i] += i;

  int err = 0;
  #pragma omp target parallel for map(tofrom:err) is_device_ptr(q)
  for(int i = 0; i < n; i++)
    if (q[i] != 2*i) {
      #pragma omp atomic
      err++;
    }

  // not freeing device memory, it will be free'ed when de-initializing the object module
  // calling omp_free requires to save omp_allocator_handle_t amongst successive target regions

  return err;
}
