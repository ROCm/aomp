#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <stdexcept>
#include <omp.h>
#include <unistd.h>

#define N 36

void check_hip_error(const std::string& func, hipError_t err)
{
  if (err != hipSuccess)
  {
    std::cout << func <<" failed" << std::endl;
    abort();
  }
}

int main()
{
  int* data = (int*)malloc(N * sizeof(int));

#if defined(USE_HIP)
  hipError_t err = hipHostRegister(data, N * sizeof(int), hipHostRegisterDefault);
  check_hip_error("hipHostRegister", err);

  int* data_dev;
  err = hipMalloc(&data_dev, N * sizeof(int));
  check_hip_error("hipMalloc", err);

  const int status = omp_target_associate_ptr(data, data_dev, N * sizeof(int), 0, omp_get_default_device());
  if (status != 0)
      throw std::runtime_error("omp_target_associate_ptr failed in OMPallocator!");
#else
  #pragma omp target enter data map(alloc: data[:N])
#endif

  #pragma omp target
  {
    data[1] = 10.0;
    printf("first touch %d\n", data[1]);
  }

  data[1] = 1.0;
  #pragma omp target update to(data[:2])

  #pragma omp target
  {
    printf("after update data %d\n", data[1]);
  }

  data[1] = 2.0;
  int* data1 = data + 1;
  #pragma omp target update to(data1[:1])
  int check = 1;
  #pragma omp target map(from:check)
  {
    if (data[1] == 2.0) check = 0;
    printf("after update data1 %d (should be 2.0)\n", data[1]);
  }
  return check;
}
