#include <stdio.h>
#include <hip/hip_runtime.h>

void hip_hostreg(double *ptr, int size) {
  hipError_t err;
  err = hipHostRegister(ptr, size, hipHostRegisterDefault);
}
