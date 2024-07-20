#include <iostream>
#include <hip/hip_runtime.h>
 
#define HIP_CHECK(stat)                                           \
{                                                                 \
    if(stat != hipSuccess)                                        \
    {                                                             \
        std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
}
 
__global__ void vector_add(double *a, double *b, double *c, int n)
{
  size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = blockDim.x * gridDim.x;
 
  for (size_t i = index; i < n; i += stride)
    c[i] = a[i] + b[i];
}
 
extern "C"
{
  void omp_hip(double *a, double *b, double *c, int N)
  {
      std::cout << "Calling HIP vector_add" << std::endl;
      vector_add<<<N/256,256>>>(a,b,c,N);
      HIP_CHECK(hipDeviceSynchronize());
      std::cout << "Finished HIP vector_add" << std::endl;
  }
}
