#include<cstdio>
#include<omp.h>

#ifndef ALIGN_VAL
#error must set ALIGN_VAL as compile time constant and default alignment value
#endif

int main() {
  omp_allocator_handle_t default_alloc = omp_get_default_allocator();
  if (default_alloc != omp_default_mem_alloc)
    return 1;

  int *p = (int *)omp_alloc(123456*sizeof(int));
  // check if it is aligned to ALIGN_VAL
  if (((uintptr_t)p) % ALIGN_VAL != 0) {
    printf("Pointer not aligned to %d\n", ALIGN_VAL);
    return 1;
  }

  printf("All good %p\n", p);
  return 0;
}
