#include<cstdio>

#pragma omp requires unified_shared_memory

int main() {
  int *a = new int[1024];

  #pragma omp target map(a[:10])
  {
    for(int i = 0; i < 10; i++)
      a[i] = i;
  }

  #pragma omp target map(a[:1024])
  {
    for(int i = 0; i < 1024; i++)
      a[i] = i;
  }

  return 0;
}
