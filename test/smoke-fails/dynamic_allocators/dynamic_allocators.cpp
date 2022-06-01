#include<cstdio>
#include<omp.h>

#pragma omp requires dynamic_allocators

int main() {
  int n = 1024;
  int *q = nullptr;
  #pragma omp target is_device_ptr(q)
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
  
  return err;
}
