#include <omp.h>
#define N 100

#pragma omp declare target
int D_Ptr[N];
#pragma omp end declare target

int main(int argc, char *argv[]) {
#pragma omp target data map(tofrom : D_Ptr[0 : N])
  {
#pragma omp target teams distribute parallel for
    for (int i = 0; i < N; i++) {
      D_Ptr[i + 1] = 2 * (i + 1);
    }
  }
  return 0;
}

/// CHECK:=================================================================
/// CHECK-NEXT:=={{[0-9]+}}==ERROR: AddressSanitizer: global-buffer-overflow on amdgpu device 0 at pc [[PC:.*]]
/// CHECK-NEXT:WRITE of size 4 in workgroup id ({{[0-9]+}},0,0)
/// CHECK-NEXT:  #0 [[PC]] in __omp_offloading_fd00_68a0014_main_l11 at {{.*}}aomp/test/smoke-asan/omp-global-buffer-overflow/omp-global-buffer-overflow.cpp:13:13
