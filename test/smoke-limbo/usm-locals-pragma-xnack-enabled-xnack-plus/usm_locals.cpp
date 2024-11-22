#include <cstdio>

// usm_locals: XNACK enabled, compiled with: gfx94X:xnack+, with USM pragma => OUTCOME: ZERO-COPY

#pragma omp requires unified_shared_memory

int main() {
  int x = 3;
  int y = -1;
  int z[10];  // 40 bytes
  int *k;     // 20 bytes

  for(size_t t = 0; t < 10; t++)
    z[t] = t;
  k = new int[5];

  printf("Host pointer for k = %p\n", k);
  for(size_t t = 0; t < 5; t++)
    k[t] = -t;

  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 40, {{.*}})
  /// CHECK-NOT: data_submit_async: {{.*}} 0 ({{.*}} 20, {{.*}})
  #pragma omp target update to(z[:10])

  #pragma omp target map(to:k[:5]) map(always, tofrom:x) map(tofrom:y)
  {
    x++;
    y++;
    for(size_t t = 0; t < 10; t++)
      z[t]++;
    printf("Device pointer for k = %p, k[3] = %d\n", k, k[3]);
  }
  #pragma omp target update from(z[:10])
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 20, {{.*}})
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 40, {{.*}})
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 4, {{.*}})
  /// CHECK-NOT: data_retrieve_async: {{.*}} 0 ({{.*}} 4, {{.*}})

  // Note: when the output is redirected rather than printed at the console,
  // the printf'd strings are printed AFTER all the OpenMP runtime library
  // messages are printed.

  /// CHECK: Host pointer for k = [[K_HOST_ADDR:0x.*]]
  /// CHECK: Device pointer for k = [[K_HOST_ADDR]], k[3] = -3
  printf("x = %d, y = %d, z[7] = %d\n", x, y, z[7]);

  delete [] k;
  return 0;
}
