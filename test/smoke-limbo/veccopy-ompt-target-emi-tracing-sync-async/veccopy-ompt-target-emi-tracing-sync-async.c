#include <assert.h>
#include <omp.h>
#include <stdio.h>

int main() {
  int M = 50;
  int N = 100000;

  int a[N], aa[M];
  int b[N], bb[M];

  int i;

  for (i = 0; i < M; i++)
    aa[i] = 0;

  for (i = 0; i < M; i++)
    bb[i] = i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

#pragma omp target parallel for
  {
    for (int j = 0; j < M; j++)
      aa[j] = bb[j];
  }

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < M; j++)
      aa[j] = bb[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < M; i++)
    if (aa[i] != bb[i]) {
      rc++;
      printf("Wrong value: aa[%d]=%d\n", i, aa[i]);
    }

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

/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_01:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_02:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_03:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_04:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_05:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_06:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_07:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_08:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_09:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_10:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_11:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_12:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_13:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_14:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_15:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_16:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_17:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_18:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_19:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_20:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_21:0x[0-f]+]] in buffer request callback
/// CHECK-DAG: Allocated {{[0-9]+}} bytes at [[ADDRX_22:0x[0-f]+]] in buffer request callback

/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_06]] {{[0-9]+}} [[ADDRX_06]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_07]] {{[0-9]+}} [[ADDRX_07]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_08]] {{[0-9]+}} [[ADDRX_08]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_09]] {{[0-9]+}} [[ADDRX_09]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_10]] {{[0-9]+}} [[ADDRX_10]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_11]] {{[0-9]+}} [[ADDRX_11]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_12]] {{[0-9]+}} [[ADDRX_12]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_13]] {{[0-9]+}} [[ADDRX_13]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_14]] {{[0-9]+}} [[ADDRX_14]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_15]] {{[0-9]+}} [[ADDRX_15]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_16]] {{[0-9]+}} [[ADDRX_16]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_17]] {{[0-9]+}} [[ADDRX_17]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_18]] {{[0-9]+}} [[ADDRX_18]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_19]] {{[0-9]+}} [[ADDRX_19]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_20]] {{[0-9]+}} [[ADDRX_20]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_21]] {{[0-9]+}} [[ADDRX_21]] {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_22]] {{[0-9]+}} [[ADDRX_22]] {{[0-9]+}}

// Note: Split checks for record address and content.
//       Since records of different target regions may occur interleaved.
/// CHECK-DAG: rec=[[ADDRX_01]]
/// CHECK-DAG: rec=[[ADDRX_02]]
/// CHECK-DAG: rec=[[ADDRX_03]]
/// CHECK-DAG: rec=[[ADDRX_04]]
/// CHECK-DAG: rec=[[ADDRX_05]]
/// CHECK-DAG: rec=[[ADDRX_06]]
/// CHECK-DAG: rec=[[ADDRX_07]]
/// CHECK-DAG: rec=[[ADDRX_08]]
/// CHECK-DAG: rec=[[ADDRX_09]]
/// CHECK-DAG: rec=[[ADDRX_10]]
/// CHECK-DAG: rec=[[ADDRX_11]]
/// CHECK-DAG: rec=[[ADDRX_12]]
/// CHECK-DAG: rec=[[ADDRX_13]]
/// CHECK-DAG: rec=[[ADDRX_14]]
/// CHECK-DAG: rec=[[ADDRX_15]]
/// CHECK-DAG: rec=[[ADDRX_16]]
/// CHECK-DAG: rec=[[ADDRX_17]]
/// CHECK-DAG: rec=[[ADDRX_18]]
/// CHECK-DAG: rec=[[ADDRX_19]]
/// CHECK-DAG: rec=[[ADDRX_20]]
/// CHECK-DAG: rec=[[ADDRX_21]]
/// CHECK-DAG: rec=[[ADDRX_22]]

/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2

/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=2
/// CHECK-DAG: type=10 (Target kernel) {{.+}} requested_num_teams=1
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=3
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=9 (Target data op) {{.+}} optype=4
/// CHECK-DAG: type=8 (Target task) {{.+}} kind=1 endpoint=2

// Note: These addresses will only occur once. They are only captured to
//       indicate their existence.
/// CHECK-DAG: rec=[[ADDRX_23:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_24:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_25:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_26:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_27:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_28:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_29:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_30:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_31:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_32:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_33:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_34:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_35:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_36:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_37:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_38:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_39:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_40:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_41:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_42:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_43:0x[0-f]+]]
/// CHECK-DAG: rec=[[ADDRX_44:0x[0-f]+]]

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

// Note: ADDRX_22 may not trigger a final callback.
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_01]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_02]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_03]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_04]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_05]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_06]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_07]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_08]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_09]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_10]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_11]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_12]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_13]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_14]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_15]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_16]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_17]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_18]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_19]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_20]] {{[0-9]+}} (nil) {{[0-9]+}}
/// CHECK-DAG: Executing buffer complete callback: {{[0-9]+}} [[ADDRX_21]] {{[0-9]+}} (nil) {{[0-9]+}}

// Note: ADDRX_22 may not be deallocated.
/// CHECK-DAG: Deallocated [[ADDRX_01]]
/// CHECK-DAG: Deallocated [[ADDRX_02]]
/// CHECK-DAG: Deallocated [[ADDRX_03]]
/// CHECK-DAG: Deallocated [[ADDRX_04]]
/// CHECK-DAG: Deallocated [[ADDRX_05]]
/// CHECK-DAG: Deallocated [[ADDRX_06]]
/// CHECK-DAG: Deallocated [[ADDRX_07]]
/// CHECK-DAG: Deallocated [[ADDRX_08]]
/// CHECK-DAG: Deallocated [[ADDRX_09]]
/// CHECK-DAG: Deallocated [[ADDRX_10]]
/// CHECK-DAG: Deallocated [[ADDRX_11]]
/// CHECK-DAG: Deallocated [[ADDRX_12]]
/// CHECK-DAG: Deallocated [[ADDRX_13]]
/// CHECK-DAG: Deallocated [[ADDRX_14]]
/// CHECK-DAG: Deallocated [[ADDRX_15]]
/// CHECK-DAG: Deallocated [[ADDRX_16]]
/// CHECK-DAG: Deallocated [[ADDRX_17]]
/// CHECK-DAG: Deallocated [[ADDRX_18]]
/// CHECK-DAG: Deallocated [[ADDRX_19]]
/// CHECK-DAG: Deallocated [[ADDRX_20]]
/// CHECK-DAG: Deallocated [[ADDRX_21]]

/// CHECK-DAG: Success

/// CHECK-NOT: rec=
/// CHECK-NOT: host_op_id=0x0
