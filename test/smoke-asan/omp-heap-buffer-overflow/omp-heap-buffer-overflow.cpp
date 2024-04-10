#include <omp.h>

int main(int argc, char *argv[]) {
  int N = 1000;
  int *Ptr = new int[N];
#pragma omp target data map(tofrom : Ptr[0 : N])
#pragma omp target teams distribute parallel for
  for (int i = 0; i < N; i++) {
    Ptr[i + 1] = 2 * (i + 1);
  }
  delete[] Ptr;
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: heap-buffer-overflow on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in __omp_offloading_fd00_68a0017_main_l7 at {{.*}}aomp/test/smoke-asan/omp-heap-buffer-overflow/omp-heap-buffer-overflow.cpp:9:11
