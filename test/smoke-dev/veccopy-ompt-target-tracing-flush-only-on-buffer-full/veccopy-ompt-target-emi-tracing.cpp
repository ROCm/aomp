/*
 * This test is run with LIBOMPTARGET_OMPT_FLUSH_ON_SHUTDOWN=false and
 * ompt_flush_trace is not invoked by the user/tool. The intention is to check
 * whether trace records are properly flushed as a buffer fills up. Currently,
 * the buffer size in callbacks.h implies that every buffer holds 2 trace
 * records. With this assumption, there should be 20 trace records. The last 2
 * trace records are not flushed because the runtime does not know yet that the
 * last buffer is full.
 */
#include <assert.h>
#include <omp.h>
#include <stdio.h>

#include "callbacks.h"

// Map of devices traced
DeviceMapPtr_t DeviceMapPtr;

int main() {
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

// clang-format off

/// CHECK-NOT: host_op_id=0x0

/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-DAG: rec=
/// CHECK-NOT: rec=

/// CHECK-DAG: Success

/// CHECK-NOT: rec=
/// CHECK-NOT: host_op_id=0x0
