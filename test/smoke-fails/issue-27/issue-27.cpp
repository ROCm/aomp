#include <stdio.h>
int main(void) {
  bool a = true;
  bool b = true;
  int * p = nullptr;
  #pragma omp target data if(a && b) map(to: p[0])
  {
      printf("%p\n", p);
  }
  return 0;
}
