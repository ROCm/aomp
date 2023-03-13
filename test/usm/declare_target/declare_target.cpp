#include <cstdio>

#pragma omp requires unified_shared_memory

#pragma omp begin declare target
int x;
#pragma omp end declare target

int main() {
  x = 3;

  printf("Host: x = %d, &x = %p\n", x, &x);

  #pragma omp target
  {
    printf("Device: x = %d, &x = %p\n", x, &x);
    x = 2;
  }

  if (x != 2) return 1;
  return 0;
}
