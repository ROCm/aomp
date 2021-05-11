#include <iostream>

#pragma omp requires unified_shared_memory

int main() {
  int *a = (int *) malloc(sizeof(int));
  bool is_set = false;

  // a is passed by-value to kernel with unified_shared_memory; set to nullptr without
  // map of is_set should be respected
  #pragma omp target map(tofrom: is_set)
  {
    if (a != nullptr)
      is_set = true;
    else
      is_set = false;
  }

  if (!is_set)
    std::cout << "a not set\n";

  return 0;
}
