#include<cstdio>
#include<omp.h>

int main() {
  int tl1, tl2;

  #pragma omp target map(from:tl1)
  {
    tl1 = omp_get_thread_limit();
  }

  #pragma omp target teams distribute parallel for map(from:tl2) thread_limit(1024)
  for(int i = 0; i < 1024; i++) {
    if (i == 0)
      tl2 = omp_get_thread_limit();
  }

  printf("tl1 = %d, tl2 = %d\n", tl1, tl2);
  if (tl1 != 256 || tl2 != 1024)
    return 1;
  return 0;
}
