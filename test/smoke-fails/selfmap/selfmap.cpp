#include <cstring>
#include <stdio.h>

#define N 100

int main() {
  int x = 0;

#pragma omp target teams distribute parallel for map(self : x)
  {
    for (unsigned i = 0; i < N; ++i)
#pragma omp atomic update
      x++;
  }

#pragma omp target teams distribute parallel for map(self, tofrom : x)
  {
    for (unsigned i = 0; i < N; ++i)
#pragma omp atomic update
      x++;
  }

  return 0;
}