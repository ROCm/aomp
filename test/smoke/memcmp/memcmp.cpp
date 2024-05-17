#include <string.h>
#include <stdlib.h>
#include <assert.h>

int main(void)
{
  int n = 128;
  int sz = n * sizeof(int);
  int* a = (int*) malloc(sz);
  int* b = (int*) malloc(sz);
  for(int i = 0; i < n; i++)
  {
    a[i] = i;
    b[i] = i;
  }

  #pragma omp target enter data map(to: a[:n], b[:n])

  int ret;
  #pragma omp target map(from:ret)
  {
    ret = memcmp(a, b, sz);
  }
 
  #pragma omp target exit data map(from: a[:n], b[:n])

  assert(ret == 0);

  return 0;
}
