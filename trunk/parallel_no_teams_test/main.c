#include <stdio.h>
#define N_CONSTANT 10

void writeIndex(int *int_array, int array_length) {
  #pragma omp target parallel for map(from:int_array[0:N_CONSTANT])
  for (int _index = 0; _index < N_CONSTANT ; ++_index)
    int_array[_index] = _index;
}

int main() {
  int hostArray[N_CONSTANT];
  for (int i = 0; i < N_CONSTANT; ++i)
    hostArray[i] = 0;

  writeIndex(hostArray,N_CONSTANT);

  int errors = 0;
  for(int i = 0; i < N_CONSTANT; ++i)
    if(hostArray[i] != i)
      errors++;

  if(errors)
    return 1;
  printf("=======  c Test passed! =======\n");
  return 0;
}
