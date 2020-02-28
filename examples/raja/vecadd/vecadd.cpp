#include <ostream>
#include <omp.h>

#include "RAJA/RAJA.hpp"

using namespace RAJA;
using namespace RAJA::statement;

const int N = 1000000;
int main()
{

  int* a = new int[N];
  int* b = new int[N];
  int* c = new int[N];

  for (int i = 0; i < N; ++i) {
    a[i] = 2*i;
    b[i] = i;
  }

#if defined(RAJA_USE_HIP)
  // HIP vector add
  int *device_a;
  int *device_b;
  int *device_c;

  hipErrchk(hipMalloc((void **)&device_a, sizeof(int) * N));
  hipErrchk(hipMalloc((void **)&device_b, sizeof(int) * N));
  hipErrchk(hipMalloc((void **)&device_c, sizeof(int) * N));

  hipErrchk(hipMemcpy( device_a, a, N * sizeof(int), hipMemcpyHostToDevice ));
  hipErrchk(hipMemcpy( device_b, b, N * sizeof(int), hipMemcpyHostToDevice ));

  RAJA::forall<RAJA::hip_exec<256>>(RAJA::RangeSegment(0, N),
    [=] RAJA_DEVICE (int i) {
    device_c[i] = device_a[i] + device_b[i];
  });

  hipErrchk(hipMemcpy( c, device_c, N * sizeof(int), hipMemcpyDeviceToHost ));

#else
  // OpenMP target vecadd
#pragma omp target data map(to: a[0:N]) map(to: b[0:N]) map(from: c[0:N]) use_device_ptr(a) use_device_ptr(b)
  RAJA::forall<RAJA::omp_target_parallel_for_exec<256>>(RAJA::RangeSegment(0, N),
    [=](int i) {
    c[i] = a[i] + b[i];
  });
#endif

  // Check output
  for(int i = 0; i < N; ++i) {
    if(c[i] != 3*i) {
      std::cout << "Error, wrong result at index " << i << " expected " << 3*i << " got " << c[i] << "\n";
      return 1;
    }
  }

  std::cout << "Success!\n";
  return 0;
}

