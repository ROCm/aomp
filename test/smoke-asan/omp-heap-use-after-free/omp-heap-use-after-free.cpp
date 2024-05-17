#include <omp.h>

int main(int argc, char *argv[]) {
  const unsigned long int N = 10000;

  float *buffer = (float *)omp_target_alloc(N * sizeof(float), 0);

  omp_target_free(buffer, 0);
#pragma omp target teams num_teams(2) is_device_ptr(buffer)
  {
#pragma omp parallel for
    for (unsigned long int i = 0; i < N; ++i) {
      buffer[i + 1] = i;
    }
  }
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-use-after-free on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in __omp_offloading_{{.*}} at {{.*}}aomp/test/smoke-asan/omp-heap-use-after-free/omp-heap-use-after-free.cpp:13:{{[0-9]+}}
