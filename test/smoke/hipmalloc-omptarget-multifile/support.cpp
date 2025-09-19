// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

#include "hip/hip_runtime.h"
#include "support.h"
#include "stdio.h"

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
