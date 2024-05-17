#include <cstdio>

int x;  

int main() {
  x = 3;

  #pragma omp target map(always, tofrom:x)
  {
    x++;
  }

  printf("x = %d\n", x );
  #pragma omp target map(always, tofrom:x)
  {
    x++;
  }

  printf("x = %d\n", x );

  return 0;
}
