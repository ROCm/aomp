// Author Tyler Allen , CLemson Univerity.
// Permission to use granted 11/28/2019
// Uses HIP managed memory from openmp target region on gfx devices.
#define __HIP_PLATFORM_HCC__ 1
#include <hip/hip_runtime.h>
#include <stdio.h>

#define SIZE    (4 * 8000000l * 128)

static double ttos(struct timespec* ts)                                                                                 
{                                                                                                                       
        return (double)ts->tv_sec + (double)ts->tv_nsec / 1000000000.0;                                                 
}   

int main(void) 
{

    float *A, *B, *C;
    hipError_t err;

    hipInit(0);
    srand(349082);
    err = hipMallocManaged(&A, SIZE * sizeof(float), hipMemAttachGlobal);
    fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, err, hipGetErrorString(err));
    err = hipMallocManaged(&B, SIZE * sizeof(float), hipMemAttachGlobal);
    fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, err, hipGetErrorString(err));
    err = hipMallocManaged(&C, SIZE * sizeof(float), hipMemAttachGlobal);
    fprintf(stderr, "[%s:%d] err[%d]: %s\n", __FILE__, __LINE__, err, hipGetErrorString(err));

    #pragma omp parallel for
    for (size_t i = 0; i < SIZE; i++) 
    {
        A[i] = (float) i;
        B[i] = (float) i * 100.0f;
        C[i] = 0.0;
    }

    fprintf(stderr, "starting \n");

    struct timespec t0;
    struct timespec t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    #pragma omp target teams distribute parallel for is_device_ptr(A,B,C)
    for (size_t i = 0; i < SIZE; i++)
    {
        C[i] = A[i] + B[i];
        if (C[i] == 1.23)               
            printf("Wowie! %f\n", C[i]);
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);                                                                                
    fprintf(stderr, "Kernel Time: %lf\n", ttos(&t1) - ttos(&t0));

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


    hipHostFree(A);
    hipHostFree(B);
    hipHostFree(C);

    return ret;
}



