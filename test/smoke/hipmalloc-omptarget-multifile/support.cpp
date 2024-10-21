#include "hip/hip_runtime.h"
#include "support.h"

void printHipError(hipError_t error) {
  fprintf(stderr,"Hip Error: %s\n", hipGetErrorString(error));
}

bool hipCallSuccessful(hipError_t error) {
  if (error != hipSuccess){
    printHipError(error);
    exit(1);
  }
  return error == hipSuccess;
}
