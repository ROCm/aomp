#include <cassert>
#include <omp.h>
#include <stdio.h>

// This test starts device tracing on the default device only (see
// start_trace in callbacks.h). However, if more devices are
// available, it calls flush and stop on the other devices as
// well. The intention is to check correct runtime behavior if a tool
// invokes flush or stop on a device that was not started. The runtime
// should just return without doing anything.

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

    // Warm up
#pragma omp target
  {}

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

  for (int dev = 0; dev < omp_get_num_devices(); ++dev) {
#pragma omp target teams distribute parallel for device(dev)
    {
      for (int j = 0; j < N; j++)
        a[j] = b[j];
    }
  }

  for (auto Dev : *DeviceMapPtr)
    flush_trace(Dev);

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

// > OMPT device tracing related checks below. <

/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_01:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_02:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_03:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_04:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_05:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_06:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_07:0x[0-f]+]] in buffer request callback

/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_01]], {{.+}} begin=[[ADDRX_01]]
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_02]], {{.+}} begin=[[ADDRX_02]]
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_03]], {{.+}} begin=[[ADDRX_03]]
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_04]], {{.+}} begin=[[ADDRX_04]]
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_05]], {{.+}} begin=[[ADDRX_05]]
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_06]], {{.+}} begin=[[ADDRX_06]]
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_07]], {{.+}} begin=[[ADDRX_07]]

// Note: Split checks for record address and content. That way we do not imply
//       any order. Records may / will occur interleaved.
/// CHECK-DAG: rec=[[ADDRX_01]]
/// CHECK-DAG: rec=[[ADDRX_02]]
/// CHECK-DAG: rec=[[ADDRX_03]]
/// CHECK-DAG: rec=[[ADDRX_04]]
/// CHECK-DAG: rec=[[ADDRX_05]]
/// CHECK-DAG: rec=[[ADDRX_06]]
/// CHECK-DAG: rec=[[ADDRX_07]]

// Note: These addresses will only occur once. They are only captured to
//       indicate their existence.
/// CHECK-DAG: rec=[[ADDRX_08:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_09:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_10:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_11:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_12:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_13:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_14:0x[0-f]+]]

/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=1
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=0
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2

// Note: ADDRX_07 may not trigger a final callback.
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_01]], {{.+}} begin=(nil)
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_02]], {{.+}} begin=(nil)
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_03]], {{.+}} begin=(nil)
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_04]], {{.+}} begin=(nil)
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_05]], {{.+}} begin=(nil)
/// CHECK-DAG: Executing buffer complete callback {{.+}} device_num=0, buffer=[[ADDRX_06]], {{.+}} begin=(nil)

// Note: ADDRX_07 may not be deallocated.
/// CHECK-DAG: Deallocated [[ADDRX_01]]
/// CHECK-DAG: Deallocated [[ADDRX_02]]
/// CHECK-DAG: Deallocated [[ADDRX_03]]
/// CHECK-DAG: Deallocated [[ADDRX_04]]
/// CHECK-DAG: Deallocated [[ADDRX_05]]
/// CHECK-DAG: Deallocated [[ADDRX_06]]

/// CHECK-NOT: rec=
/// CHECK-NOT: host_op_id=0x0
