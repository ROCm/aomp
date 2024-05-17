// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hip/hip_runtime.h"
#include <stdio.h>

#define N 3
#define M 3
#define P 3
#define Z 10

__global__ void matrixMul(int *matrixA, int *matrixB, int *matrixC, int ARows,
                          int ACols, int BCols) {
  int i = hipBlockIdx_x;
  int j = hipBlockIdx_y;

  if (i < ARows && j < BCols) {
    int value = 0;
    for (int k = 0; k < ACols; ++k) {
      value += matrixA[i * ACols + k] * matrixB[k * BCols + j];
    }
    matrixC[i * BCols + j] = value;
  }
}

int matrixMul_check(int *matrixA, int *matrixB, int *matrixC, int ARows,
                     int ACols, int BCols) {
  int n_errors = 0;
  for (int j = 0; j < BCols; ++j) {
    for (int i = 0; i < ARows; ++i) {
      int value = 0;
      for (int k = 0; k < ACols; ++k) {
        value += matrixA[i * ACols + k] * matrixB[k * BCols + j];
      }
      if(matrixC[i * BCols + j] != value) {
        ++n_errors;
        fprintf(stderr,"\tError: Matrices miscompare devC[%d,%d]-hostC: %d\n", i, j,
               matrixC[i * BCols + j] - value);
      }
    }
  }
  return n_errors;
}

void printMatrix(int *matrix, int Rows, int Cols) {
  for (int i = 0; i < Rows; ++i) {
    fprintf(stderr,"\n[");
    bool first = true;
    for (int j = 0; j < Cols; ++j) {
      if (first) {
        fprintf(stderr,"%d", matrix[i * Cols + j]);
        first = false;
      } else {
        fprintf(stderr,", %d", matrix[i * Cols + j]);
      }
    }
    fprintf(stderr,"]");
  }
}

void printHipError(hipError_t error) {
  fprintf(stderr,"Hip Error: %s\n", hipGetErrorString(error));
}

void randomizeMatrix(int *matrix, int Rows, int Cols) {
  for (int i = 0; i < Rows * Cols; ++i)
    matrix[i] = rand() % 10;
}

void clearMatrix(int *matrix, int Rows, int Cols) {
  for (int i = 0; i < Rows * Cols; ++i)
    matrix[i] = 0;
}
bool hipCallSuccessful(hipError_t error) {
  if (error != hipSuccess)
    printHipError(error);
  return error == hipSuccess;
}

bool deviceCanCompute(int deviceID) {
  bool canCompute = false;
  hipDeviceProp_t deviceProp;
  bool devicePropIsAvailable =
      hipCallSuccessful(hipGetDeviceProperties(&deviceProp, deviceID));
  if (devicePropIsAvailable) {
    canCompute = deviceProp.computeMode != hipComputeModeProhibited;
    if (!canCompute)
      fprintf(stderr,"Compute mode is prohibited\n");
  }
  return canCompute;
}

bool deviceIsAvailable(int *deviceID) {
  return hipCallSuccessful(hipGetDevice(deviceID));
}

// We always use device 0
bool haveComputeDevice() {
  int deviceID = 0;
  return deviceIsAvailable(&deviceID) && deviceCanCompute(deviceID);
}

int main() {
  int N_errors = 0;
  if (!haveComputeDevice()) {
    fprintf(stderr,"No compute device available\n");
    return 0;
  }
#pragma omp parallel for schedule(static,1)
  for(int i=0; i<Z; ++i) {
#pragma omp task
   {
    fprintf(stderr,"Interation: %d started <<<<\n",i);
    int hostSrcMatA[N * M];
    int hostSrcMatB[M * P];
    int hostDstMat[N * P];

    randomizeMatrix(hostSrcMatA, N, M);
    randomizeMatrix(hostSrcMatB, M, P);
    clearMatrix(hostDstMat, N, P);

//    fprintf(stderr,"A: ");
//    printMatrix(hostSrcMatA, N, M);
//    fprintf(stderr,"\nB: ");
//    printMatrix(hostSrcMatB, M, P);
//    fprintf(stderr,"\n");

    int *deviceSrcMatA = NULL;
    int *deviceSrcMatB = NULL;
    int *deviceDstMat = NULL;

    bool matrixAAllocated = hipCallSuccessful(
        hipMalloc((void **)&deviceSrcMatA, N * M * sizeof(int)));
    bool matrixBAllocated = hipCallSuccessful(
        hipMalloc((void **)&deviceSrcMatB, M * P * sizeof(int)));
    bool matrixCAllocated =
        hipCallSuccessful(hipMalloc((void **)&deviceDstMat, N * P * sizeof(int)));

    if (matrixAAllocated && matrixBAllocated && matrixCAllocated) {
      bool copiedSrcMatA=false, copiedSrcMatB=false;
//#pragma omp task shared(copiedSrcMatA)
      copiedSrcMatA = hipCallSuccessful(hipMemcpy(deviceSrcMatA, hostSrcMatA,
                                                     N * M * sizeof(int),
                                                     hipMemcpyHostToDevice));
//#pragma omp task shared(copiedSrcMatB)
      copiedSrcMatB = hipCallSuccessful(hipMemcpy(deviceSrcMatB, hostSrcMatB,
                                                     M * P * sizeof(int),
                                                     hipMemcpyHostToDevice));
//#pragma omp taskwait
      if (copiedSrcMatA && copiedSrcMatB) {
        dim3 dimGrid(N, P);
        matrixMul<<<dimGrid, 1, 0, 0>>>(deviceSrcMatA, deviceSrcMatB,
                                        deviceDstMat, N, M, P);
        if (hipCallSuccessful(hipMemcpy(hostDstMat, deviceDstMat,
                                        N * P * sizeof(int),
                                        hipMemcpyDeviceToHost))) {

          N_errors = matrixMul_check(hostSrcMatA, hostSrcMatB, hostDstMat,
                                         N, M, P);
          if (N_errors != 0) {
            fprintf(stderr,"Interation: %d \t\t FAILED: %d Errors >>>\n", i, N_errors);
            fprintf(stderr,"A: ");
            printMatrix(hostSrcMatA, N, M);
            fprintf(stderr,"\nB: ");
            printMatrix(hostSrcMatB, M, P);
            fprintf(stderr,"\n");
            fprintf(stderr,"Mul: ");
            printMatrix(hostDstMat, N, P);
            fprintf(stderr,"\n");
          } else fprintf(stderr,"Interation: %d \t\t SUCCESSFUL >>>\n", i);
        } else {
          fprintf(stderr,"Unable to copy memory from device to host\n");
        }
      } else fprintf(stderr,"Unable to make initial copy\n");
    }
//#pragma omp task shared(matrixAAllocated, deviceSrcMatA)
    if (matrixAAllocated)
      hipFree(deviceSrcMatA);
//#pragma omp task shared(matrixBAllocated, deviceSrcMatB)
    if (matrixBAllocated)
      hipFree(deviceSrcMatB);
//#pragma omp task shared(matrixCAllocated, deviceDstMat)
    if (matrixCAllocated)
      hipFree(deviceDstMat);
//#pragma omp taskwait
    fprintf(stderr,"Interation: %d finished >>>\n",i);
   }
  }
  if(!N_errors)
    fprintf(stderr,"%s" , "Success");
  else
    fprintf(stderr,"%s", "Failed\n");

  return N_errors;
}
