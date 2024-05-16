#include <iostream>

#ifdef REQUIRES_USM
#pragma omp requires unified_shared_memory
#define IS_USM 1
#else
#define IS_USM 0
#endif

int main() {
  int *a = (int *) malloc(sizeof(int));
  int is_set = 1;
  printf("host   a:%p \n", (void*) a);

  // a is passed by-value to kernel with unified_shared_memory; set to nullptr without
  // map of is_set should be respected
  #pragma omp target map(tofrom: is_set)
  {
    printf("device a:%p \n", (void*) a);
    if (a == nullptr)
      is_set = 0;
    else
      is_set = 1;
  }

  if (is_set)
    std::cout << "Pointer a is set. This should be USM mode\n";
  else
    std::cout << "Pointer a is NOT set. This should be default mode \n";

 if (IS_USM && is_set) 
   return 0;
 else if (!IS_USM && !is_set) 
   return 0;
 else if (!IS_USM && is_set) 
   return 1;
 else if (IS_USM && !is_set) 
   return 1;
}
