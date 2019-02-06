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

#define N 10

__global__
void addVector(int *vectorA, int *vectorB, int*vectorC)
{
  int i = blockIdx.x;
  if (i<N) {
    vectorC[i] = vectorA[i] + vectorB[i];
  }
}

void printVector(int *vector)
{
  printf("[");
  bool first = true;
  for (int i = 0; i<N; ++i)
  {
    if (first)
    {
      printf("%d", vector[i]);
      first = false;
    }
    else
    {
      printf(", %d", vector[i]);
    }
  }
  printf("]");
}

void printCudaError(cudaError_t error)
{
  printf("Cuda Error: %s\n", cudaGetErrorString(error));
}

void randomizeVector(int *vector)
{
  for (int i = 0; i < N; ++i)
    vector[i] = rand() % 10;
}

void clearVector(int *vector)
{
  for (int i = 0; i < N; ++i)
    vector[i] = 0;
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
  int hostSrcVecA[N];
  int hostSrcVecB[N];
  int hostDstVec[N];

  if (!haveComputeDevice())
  {
    printf("No compute device available\n");
    return 0;
  }

  randomizeVector(hostSrcVecA);
  randomizeVector(hostSrcVecB);
  clearVector(hostDstVec);

  printf("  A: ");
  printVector(hostSrcVecA);
  printf("\n  B: ");
  printVector(hostSrcVecB);
  printf("\n");

  int *deviceSrcVecA = NULL;
  int *deviceSrcVecB = NULL;
  int *deviceDstVec = NULL;

  bool vectorAAllocated =
    cudaCallSuccessful(cudaMalloc((void **)&deviceSrcVecA, N*sizeof(int)));
  bool vectorBAllocated =
    cudaCallSuccessful(cudaMalloc((void **)&deviceSrcVecB, N*sizeof(int)));
  bool vectorCAllocated =
    cudaCallSuccessful(cudaMalloc((void **)&deviceDstVec, N*sizeof(int)));

  if (vectorAAllocated && vectorBAllocated && vectorCAllocated)
  {
    bool copiedSrcVecA =
      cudaCallSuccessful(cudaMemcpy(deviceSrcVecA, hostSrcVecA,
                                    N * sizeof(int), cudaMemcpyHostToDevice));
    bool copiedSrcVecB =
      cudaCallSuccessful(cudaMemcpy(deviceSrcVecB, hostSrcVecB,
                                    N * sizeof(int), cudaMemcpyHostToDevice));

    if (copiedSrcVecA && copiedSrcVecB)
    {
      addVector<<<N, 1>>>(deviceSrcVecA, deviceSrcVecB, deviceDstVec);

      if (cudaCallSuccessful(cudaMemcpy(hostDstVec,
                                        deviceDstVec,
                                        N * sizeof(int),
                                        cudaMemcpyDeviceToHost)))
      {
        printf("Sum: ");
        printVector(hostDstVec);
        printf("\n");
      }
      else
      {
        printf("Unable to copy memory from device to host\n");
      }
    }
  }

  if (vectorAAllocated)
    cudaFree(deviceSrcVecA);
  if (vectorBAllocated)
    cudaFree(deviceSrcVecB);
  if (vectorCAllocated)
    cudaFree(deviceDstVec);

  return 0;
}
