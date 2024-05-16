#include<cstdio>

#pragma omp requires unified_shared_memory

#define N 1024

struct T {
  int x, y, z;
};

int main() {
  int A[N], B[N], C[N];
  T t; t.x = 1; t.y = 2; t.z = 3;
  int red = 0;

  for(int i = 0; i < N; i++) {
    B[i] = i;
    C[i] = i+3;
  }

  // CHECK-COARSE: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-COARSE: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-COARSE: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-COARSE: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-FINE-NOT: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-FINE-NOT: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-FINE-NOT: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-FINE-NOT: tgt_rtl_set_coarse_grain_mem_region
  // CHECK: tgt_rtl_launch_kernel
  #pragma omp target teams distribute parallel for
  for(int i = 0; i < N; i++) {
    A[i] = B[i] + C[i];
    if (i == 0) t.x += t.y+t.z;
  }

  // CHECK-COARSE: tgt_rtl_set_coarse_grain_mem_region
  // CHECK-FINE-NOT: tgt_rtl_set_coarse_grain_mem_region
  // CHECK: tgt_rtl_launch_kernel
  #pragma omp target teams distribute parallel for reduction(+:red)
  for(int i = 0; i < N; i++)
    red += i;

  int err = 0;
  if (red != ((N-1)*N/2)) {
    err++;
    printf("reduction error: got %d expected %d\n", red, ((N-1)*N/2));
  }
  if (t.x != 6) {
    err++;
    printf("t was not updated: got %d expected %d\n", t.x, 6);
  }
  for(int i = 0; i < N; i++)
    if (A[i] != B[i] + C[i]) {
      err++;
      printf("err at %d: got %d, expected %d\n", i, A[i], B[i]+C[i]);
      if (err > 10) return err;
    }

  return err;
}
