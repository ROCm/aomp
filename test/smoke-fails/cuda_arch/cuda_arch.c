// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier:  MIT

//#include <stdio.h>

int main(int argc, char **argv) {
#ifdef __CUDA_ARCH__
#error CUDA_ARCH is set!!
#endif
  return 0;
}
