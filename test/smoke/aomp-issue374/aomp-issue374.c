/* Based on https://github.com/ROCm-Developer-Tools/aomp/issues/374 */

#include <stdio.h>
#include <assert.h>
#include <omp.h>

#include "callbacks.h"

int main()
{
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

  start_trace();
  
  int initial_device = omp_get_initial_device();
  printf("initial device_num = %d\n", initial_device);
#pragma omp target data map(to: i)  device(0)
  {
#pragma omp target device(0)
    {
      int device_num = omp_get_device_num();
      printf("device_num = %d\n", device_num);
      for (int j = 0; j< N; j++)
        a[j]=b[j];
    }
  }
  
  stop_trace();

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: Callback Target
/// CHECK-SAME: device_num=0
/// CHECK: Init: device_num=0
/// CHECK: Load: device_num:0
/// CHECK: Callback DataOp
/// CHECK-SAME: optype=1
/// CHECK-NOT: src_device_num=0
/// CHECK-SAME: dest_device_num=0 bytes=4
/// CHECK: Callback DataOp
/// CHECK-SAME: optype=2
/// CHECK-NOT: src_device_num=0
/// CHECK-SAME: dest_device_num=0 bytes=4
/// CHECK: Callback DataOp
/// CHECK-SAME: optype=1
/// CHECK-NOT: src_device_num=0
/// CHECK-SAME: dest_device_num=0 bytes=400000
/// CHECK: Callback DataOp
/// CHECK-SAME: optype=2
/// CHECK-NOT: src_device_num=0
/// CHECK-SAME: dest_device_num=0 bytes=400000
/// CHECK: optype=3
/// CHECK-SAME: src_device_num=0
/// CHECK-NOT: dest_device_num=0
/// CHECK-SAME: bytes=400000
/// CHECK: optype=4
/// CHECK-SAME: src_device_num=0
/// CHECK-SAME: dest_device_num=-1
/// CHECK-SAME: bytes=0





