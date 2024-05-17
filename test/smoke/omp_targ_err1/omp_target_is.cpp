#include <iostream>

int main() {
  int *a = (int *) malloc(2*sizeof(int));
  a[0] = 1;
  a[1] = 2;

#pragma omp target map(to: a[0:1])
  {
    printf("DEVICE : a:%p &a[1]:%p a[0]:%d  a[1]:%d\n", (void*) a, &a[1],a[0],a[1]);
  }
  return 0;
}

