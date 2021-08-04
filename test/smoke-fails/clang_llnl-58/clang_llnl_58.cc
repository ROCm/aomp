#include "hip/hip_runtime.h"

int main()
{
    hipError_t err;
    double* dummyfirst;

    err=hipMalloc(&dummyfirst, sizeof(double)*4);

    return (0);
}
