#include <iostream>
#include <omp.h>

#pragma omp requires unified_shared_memory
#define IS_USM 1

int main() {
  int *a = (int *) malloc(2*sizeof(int));
  a[0] = 1;
  a[1] = 2;
#if IS_USM >= 1
      printf("\n ---- THIS IS USM ----\n");
#endif
  int is_set = 1;
  int rc_a,rc_b;
  rc_a = omp_target_is_present(a,omp_get_default_device());
  rc_b = omp_target_is_present(&a[1],omp_get_default_device());
  fprintf(stderr," a0,a1 is present before target data region   : %d,%d \n",rc_a,rc_b);
#pragma omp target enter data map(to: a[0:1])
  fprintf(stderr,"HOST   : a:%p &a[1]:%p a[0]:%d  a[1]:%d\n", (void*) a, &a[1],a[0],a[1]);
  rc_a = omp_target_is_present(a,omp_get_default_device());
  rc_b = omp_target_is_present(&a[1],omp_get_default_device());
  fprintf(stderr," a0,a1 is present before target region        : %d,%d \n",rc_a,rc_b);
#pragma omp target map(tofrom: is_set) is_device_ptr(stderr)
  {
#if IS_USM >= 1
// We use proprocessing macro, because cannot yet reference data in USM mode, even in an unused printf
    fprintf(stderr,"DEVICE : a:%p &a[1]:%p No VALUES in USM MODE!\n", (void*) a,(void*) &a[1]);
#else
    fprintf(stderr,"DEVICE : a:%p &a[1]:%p a[0]:%d  a[1]:%d\n", (void*) a, &a[1],a[0],a[1]);
#endif
    if (a == nullptr)
      is_set = 0;
    else
      is_set = 1;
  }
  rc_a = omp_target_is_present(a,omp_get_default_device());
  rc_b = omp_target_is_present(&a[1],omp_get_default_device());
  fprintf(stderr," a0,a1 is present inside target data region   : %d,%d \n",rc_a,rc_b);
#pragma omp target exit data map(delete:a[0:1])
  rc_a = omp_target_is_present(a,omp_get_default_device());
  rc_b = omp_target_is_present(&a[1],omp_get_default_device());
  fprintf(stderr," a0,a1 is present outside target data region  : %d,%d \n",rc_a,rc_b);

  if (is_set)
    std::cout << "Pointer a is set. Either USM mode or a is mapped\n";
  else
    std::cout << "Pointer a is NOT set. " ;

 if (IS_USM && is_set) 
   return 0;
 else if (!IS_USM && !is_set) 
   return 0;
 else if (!IS_USM && is_set) 
   return 1;
 else if (IS_USM && !is_set) 
   return 1;
}
