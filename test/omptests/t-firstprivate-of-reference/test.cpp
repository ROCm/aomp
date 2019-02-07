#include <stdio.h>

int main() {

  int a = 0, b = 0, x = 1;
  bool onDevice = true;
  int max_threads = 10;

#pragma omp target map(tofrom : a, b, x) map(to : max_threads) if (onDevice)
#pragma omp parallel num_threads(max_threads)
#pragma omp sections firstprivate(x) lastprivate(x)
  {
#pragma omp section
    {
      a = 0 + x;
      x = 2;
    }
#pragma omp section 
    { 
      b = 1 + x;
      x = 3;
    }
  }

  printf("%d %d %d\n", a, b, x);

  return 0;
}