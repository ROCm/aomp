/*
 * MIT License
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE. 
 */

// Author Tyler Allen , CLemson Univerity.
// Permission to use granted 11/28/2019
// Uses HIP managed memory from openmp target region on gfx devices.

// Additional notes:
// Modified to use malloc if compiled with -DUSE_MALLOC
//    make run USE_MALLOC=-DUSE_MALLOC=1
// Reduced the size so it fits on my GFX900 which has 8GB
// compile with -DLARGE to increase memory to  16x 256,000,000
//    make run LARGE=-DLARGE=1
// Added iterations control, deafults to 1
//    make run ITERS=-DITERS=3
// Managed memory appears to allow aapping larger memory than what
// is available on the GPU.
//    make run USE_MALLOC=-DUSE_MALLOC=1 LARGE=-DLARGE=1
//    will fail to allocate enough memory on GFX900 8GB
// Increaseing iteration allows comparing cost of comm vs comp.
//    make run USE_MALLOC=-DUSE_MALLOC=1  ITERS=-DITERS=50
//    make run ITERS=-DITERS=50

// Added additional timer steps, to see cost of memory allocation, movement.

#if USE_MALLOC
#include <stdlib.h>
#endif
#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>
#include <stdio.h>

#if LARGE
#define SIZE    (4 * 8000000l * 128)
#else
#define SIZE    ( 8000000l * 32 )
#endif

#ifndef ITERS
#define ITERS 1
#endif

#pragma omp requires unified_shared_memory
static double ttos(struct timespec* ts) {
  return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;
}

int main(void)
{
    float *A, *B, *C;
    hipError_t err;

    struct timespec t0, t1, t2, t3;
    clock_gettime(CLOCK_MONOTONIC, &t0);
#if USE_MALLOC
    A = (float*) malloc(SIZE * sizeof(float));
    if(!A) fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, errno, strerror(errno));
    B = (float*) malloc(SIZE * sizeof(float));
    if (!B) fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, errno, strerror(errno));
    C = (float*) malloc(SIZE * sizeof(float));
    if(!C) fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, errno, strerror(errno));
#else
    hipInit(0);
    err = hipMallocManaged(&A, SIZE * sizeof(float), hipMemAttachGlobal);
    if (err) fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, err, hipGetErrorString(err));
    err = hipMallocManaged(&B, SIZE * sizeof(float), hipMemAttachGlobal);
    if (err) fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, err, hipGetErrorString(err));
    err = hipMallocManaged(&C, SIZE * sizeof(float), hipMemAttachGlobal);
    if (err) fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, err, hipGetErrorString(err));
#endif
    fprintf(stderr, "A:%p B:%p C:%p\n", (void*)A, (void*)B, (void*)C);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "Memory alloc time: %lf\n", ttos(&t1) - ttos(&t0));

    clock_gettime(CLOCK_MONOTONIC, &t1);
    #pragma omp parallel for
    for (size_t i = 0; i < SIZE; i++)
    {
        A[i] = (float) i;
        B[i] = (float) i * 100.0f;
        C[i] = 0.0;
    }
    clock_gettime(CLOCK_MONOTONIC, &t2);
    fprintf(stderr, "Data init time: %lf\n", ttos(&t2) - ttos(&t1));

    fprintf(stderr, "starting \n");


    clock_gettime(CLOCK_MONOTONIC, &t2);
#if USE_MALLOC

    #pragma omp target enter data map(to:A[0:SIZE],B[0:SIZE],C[0:SIZE])
    clock_gettime(CLOCK_MONOTONIC, &t3);
    fprintf(stderr, "MAP  To Time: %lf\n", ttos(&t3) - ttos(&t2));
    clock_gettime(CLOCK_MONOTONIC, &t2);

    #pragma omp target teams distribute parallel for
#else
    #pragma omp target teams distribute parallel for is_device_ptr(A,B,C)
#endif
    for (size_t i = 0; i < SIZE; i++)
    {
      for (int j=0; j < ITERS; j++) {
        C[i] = A[i] + B[i];
      }
    }

    clock_gettime(CLOCK_MONOTONIC, &t3);
    fprintf(stderr, "Kernel Time: %lf\n", ttos(&t3) - ttos(&t2));

#if USE_MALLOC
    clock_gettime(CLOCK_MONOTONIC, &t2);
    #pragma omp target exit data map(from:C[0:SIZE])
    clock_gettime(CLOCK_MONOTONIC, &t3);
    fprintf(stderr, "MAP From Time: %lf\n", ttos(&t3) - ttos(&t2));
#endif
    fprintf(stderr, "%d\n", hipGetLastError());
    int ret = 0;
    #pragma omp parallel for
    for (size_t i = 0; i < SIZE; i++)
    {
        float sol = i + i * 100.0f;
        if (C[i] != sol)
        {
            fprintf(stderr, "Error at index %ld: expected %f != %f\n", i, C[i], sol);
            ret = 1;
        }
    }
    fprintf(stderr, "Total Time: %lf\n", ttos(&t3) - ttos(&t0));
#if USE_MALLOC
    free(A);
    free(B);
    free(C);
#else
    hipHostFree(A);
    hipHostFree(B);
    hipHostFree(C);
#endif
    return ret;
}



