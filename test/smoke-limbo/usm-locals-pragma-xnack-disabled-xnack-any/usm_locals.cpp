#include <cstdio>

// usm_locals: XNACK disabled, compiled with: gfxXXX, with USM pragma => OUTCOME: ZERO-COPY (+warning)

#pragma omp requires unified_shared_memory

int main() {
  int x = 3;
  int y = -1;
  int z[10];  // 40 bytes
  int *k;     // 20 bytes

  for(size_t t = 0; t < 10; t++)
    z[t] = t;
  k = new int[5];

  printf("Host pointer for z = %p\n", z);
  printf("Host pointer for k = %p\n", k);
  for(size_t t = 0; t < 5; t++)
    k[t] = -t;

  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 40, {{.*}})
  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 20, {{.*}})
  #pragma omp target update to(z[:10])

  #pragma omp target map(tofrom:k[:5]) map(tofrom:y)
  {
    printf("Device pointer for z = %p\n", z);
    printf("Device pointer for k = %p\n", k);
    // No access to y, z, k possible.
    x++;
  }
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 20, {{.*}})
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 40, {{.*}})
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  #pragma omp target update from(z[:10])

  /// CHECK: AMDGPU message: Running a program that requires XNACK on a system where XNACK is disabled. This may cause problems when using an OS-allocated pointer inside a target region. Re-run with HSA_XNACK=1 to remove this warning.

  // Note: when the output is redirected rather than printed at the console,
  // the printf'd strings are printed AFTER all the OpenMP runtime library
  // messages are printed.

  /// CHECK: Host pointer for z = [[Z_HOST_ADDR:0x.*]]
  /// CHECK: Host pointer for k = [[K_HOST_ADDR:0x.*]]

  // Device pointer cannot be the same as the host pointer as USM is not
  // enabled.
  /// CHECK: Device pointer for z = [[Z_HOST_ADDR]]
  /// CHECK: Device pointer for k = [[K_HOST_ADDR]]
  // Note: Update to x not visible outside target region:
  printf("x = %d, y = %d, z[7] = %d\n", x, y, z[7]);

  delete [] k;
  return 0;
}
