#include <cstdio>
#include "hip/hip_runtime.h"
#include "support.h"

#define SIZE 4

int hipMallocOmpTarget1() {
    int *array = NULL;
    int finalArray[SIZE];

    hipCallSuccessful(hipMalloc((void **)&array, SIZE *sizeof(int)));
    int error = 0;

    printf("\nOutput from hipMallocOmpTarget1:\n");
    #pragma omp target is_device_ptr(array) map(from: finalArray)
      for (int i = 0; i < SIZE; ++i) {
        array[i] = i;
	finalArray[i] = i;
        printf("hipMalloc memory in omp target region: array[%i] = %i\n", i, array[i]);
      }

    hipCallSuccessful(hipFree(array));

    for (int i = 0; i < SIZE; ++i) {
      printf("Final Host Array: finalArray[%i] = %i\n",  i, finalArray[i]);
      if (finalArray[i] != i){
        printf("Error at finalArray[%i]: %i != %i\n", i, finalArray[i], i);
	return 1;
      }
    }
    return 0;
}
