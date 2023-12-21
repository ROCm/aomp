#include <stdio.h>
#define N_CONSTANT 10000

void writeIndex(int *int_array) {
  for (int _index = 0; _index < N_CONSTANT ; ++_index)
    int_array[_index] = _index;
}

int computeSum(int *int_array) {
  int sum = 1000;
#pragma omp target teams distribute parallel for map(tofrom:int_array[0:N_CONSTANT]) map(tofrom: sum) reduction(+:sum)
  for (int _index = 0; _index < N_CONSTANT ; ++_index)
    sum += int_array[_index];

  return sum;
}

int main() {
  int hostArray[N_CONSTANT];
  for (int i = 0; i < N_CONSTANT; ++i)
    hostArray[i] = 0;

  writeIndex(hostArray);

  int errors = 0;
  int hostSum = 1000;
  for(int i = 0; i < N_CONSTANT; ++i) {
    hostSum += hostArray[i];
    if(hostArray[i] != i)
      errors++;
  }

  int deviceSum = computeSum(hostArray);
  if(deviceSum != hostSum)
    errors++;

  if(errors)
    return 1;


  printf("=======  c Test passed! =======\n");
  return 0;
}
