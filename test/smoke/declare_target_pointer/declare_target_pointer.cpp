//From AOMP issue #198
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#pragma omp declare target
int * arr;
#pragma omp end declare target

int main(void)
{
  // Allocate array and set to zero
  int len = 6;
  arr = (int *) calloc( len, sizeof(int) );

  // Map and fill with values on device
  #pragma omp target teams distribute parallel for map(tofrom: arr[:len])
  for( int i = 0; i < len; i++)
  {
    arr[i] = i;
  }

  assert(arr[5] == 5 && "Results are Incorrect");

  return 0;
}
