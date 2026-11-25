#include <cstdio>
#include <omp.h>
#include <unistd.h>

#define N 123456

int main() {
  int err = 0;
  int n = N;

  // allocator for locked memory with predefined allocator
  double *a = (double *)omp_alloc(n * sizeof(double), ompx_pinned_mem_alloc);
  double *b = (double *)omp_alloc(n * sizeof(double), ompx_pinned_mem_alloc);

  for (int i = 0; i < n; i++) {
    a[i] = 0;
    b[i] = i;
  }

#pragma omp target teams distribute parallel for map(to : b[ : n])             \
    map(from : a[ : n])
  for (int i = 0; i < n; i++) {
    a[i] = b[i];
  }
  //  sleep(5);
  for (int i = 0; i < n; i++)
    if (a[i] != b[i]) {
      err++;
      printf("Error at %d, expected %lf, got %lf\n", i, b[i], a[i]);
      if (err > 10)
        return err;
    }

  omp_free(a, ompx_pinned_mem_alloc);
  omp_free(b, ompx_pinned_mem_alloc);

  if (!err)
    printf("Success\n");

  return err;
}

// clang-format off

/// CHECK-NOT: host_op_id=0x0

/// CHECK: 0: Could not register callback 'ompt_callback_target_map_emi'

/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_01:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_02:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_03:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_04:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_05:0x[0-f]+]] in buffer request callback

/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=0 granted_num_teams=416
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2

/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}}

/// CHECK-DAG: Success

/// CHECK-NOT: rec=
/// CHECK-NOT: host_op_id=0x0
