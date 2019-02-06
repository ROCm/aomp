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

#include <stdio.h>

#define N 3
#define M 3
#define P 3

__global__
void matrixMul(int *matrixA, int *matrixB, int *matrixC,
               int ARows, int ACols, int BCols )
{
  int i = blockIdx.x;
  int j = blockIdx.y;

  if (i < ARows && j < BCols)
  {
    int value = 0;
    for (int k = 0; k < ACols; ++k)
    {
      value += matrixA[i*ACols+k] * matrixB[k*BCols+j];
    }
    matrixC[i*BCols+j] = value;
  }
}

void printMatrix(int *matrix, int Rows, int Cols)
{
  for (int i = 0; i < Rows; ++i)
  {
    printf("\n[");
    bool first = true;
    for (int j = 0; j < Cols; ++j)
    {
      if (first)
      {
        printf("%d", matrix[i*Cols+j]);
        first = false;
      }
      else
      {
        printf(", %d", matrix[i*Cols+j]);
      }
    }
    printf("]");
  }
}

void printCudaError(cudaError_t error)
{
  printf("Cuda Error: %s\n", cudaGetErrorString(error));
}

void randomizeMatrix(int *matrix, int Rows, int Cols)
{
  for (int i = 0; i < Rows*Cols; ++i)
    matrix[i] = rand() % 10;
}

void clearMatrix(int *matrix, int Rows, int Cols )
{
  for (int i = 0; i < Rows*Cols; ++i)
    matrix[i] = 0;
}
bool cudaCallSuccessful(cudaError_t error)
{
  if (error != cudaSuccess)
    printCudaError(error);
  return error == cudaSuccess;
}

bool deviceCanCompute(int deviceID)
{
  bool canCompute = false;
  cudaDeviceProp deviceProp;
  bool devicePropIsAvailable =
    cudaCallSuccessful(cudaGetDeviceProperties(&deviceProp, deviceID));
  if (devicePropIsAvailable)
  {
    canCompute = deviceProp.computeMode != cudaComputeModeProhibited;
    if (!canCompute)
      printf("Compute mode is prohibited\n");
  }
  return canCompute;
}

bool deviceIsAvailable(int *deviceID)
{
  return cudaCallSuccessful(cudaGetDevice(deviceID));
}

// We always use device 0
bool haveComputeDevice()
{
  int deviceID = 0;
  return deviceIsAvailable(&deviceID) && deviceCanCompute(deviceID);
}

int main()
{
  int hostSrcMatA[N*M];
  int hostSrcMatB[M*P];
  int hostDstMat[N*P];

  if (!haveComputeDevice())
  {
    printf("No compute device available\n");
    return 0;
  }

  randomizeMatrix(hostSrcMatA, N, M);
  randomizeMatrix(hostSrcMatB, M, P);
  clearMatrix(hostDstMat, N, P);

  printf("A: ");
  printMatrix(hostSrcMatA, N, M);
  printf("\nB: ");
  printMatrix(hostSrcMatB, M ,P);
  printf("\n");

  int *deviceSrcMatA = NULL;
  int *deviceSrcMatB = NULL;
  int *deviceDstMat = NULL;

  bool matrixAAllocated =
    cudaCallSuccessful(cudaMalloc((void **)&deviceSrcMatA, N*M*sizeof(int)));
  bool matrixBAllocated =
    cudaCallSuccessful(cudaMalloc((void **)&deviceSrcMatB, M*P*sizeof(int)));
  bool matrixCAllocated =
    cudaCallSuccessful(cudaMalloc((void **)&deviceDstMat, N*P*sizeof(int)));

  if (matrixAAllocated && matrixBAllocated && matrixCAllocated)
  {
    bool copiedSrcMatA =
      cudaCallSuccessful(cudaMemcpy(deviceSrcMatA, hostSrcMatA,
                                    N*M*sizeof(int),
                                    cudaMemcpyHostToDevice));
    bool copiedSrcMatB =
      cudaCallSuccessful(cudaMemcpy(deviceSrcMatB, hostSrcMatB,
                                    M*P*sizeof(int),
                                    cudaMemcpyHostToDevice));

    if (copiedSrcMatA && copiedSrcMatB)
    {
        dim3 dimGrid(N,P);
        matrixMul<<<dimGrid, 1>>>(deviceSrcMatA, deviceSrcMatB, deviceDstMat,
                                  N, M, P);
      if (cudaCallSuccessful(cudaMemcpy(hostDstMat,
                                        deviceDstMat,
                                        N*P*sizeof(int),
                                        cudaMemcpyDeviceToHost)))
      {
        printf("Mul: ");
        printMatrix(hostDstMat, N, P);
        printf("\n");
      }
      else
      {
        printf("Unable to copy memory from device to host\n");
      }
    }
  }

  if (matrixAAllocated)
    cudaFree(deviceSrcMatA);
  if (matrixBAllocated)
    cudaFree(deviceSrcMatB);
  if (matrixCAllocated)
    cudaFree(deviceDstMat);

  return 0;
}
