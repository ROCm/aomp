#include <stdio.h>
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

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

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

  /// CHECK: Callback Init:
  /// CHECK: Callback Load:
  /// CHECK: Callback Target EMI: kind=1 endpoint=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback Submit: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] req_num_teams=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback Target EMI: kind=1 endpoint=2
  /// CHECK: Callback Target EMI: kind=1 endpoint=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=1
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=2
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=2
  /// CHECK: Callback Submit: target_id=[[TARGET_ID:[0-9]+]] host_op_id=[[HOST_OP_ID:[0-9]+]] req_num_teams=0
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=3
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=1 optype=4
  /// CHECK: Callback DataOp EMI: endpoint=2 optype=4
  /// CHECK: Callback Target EMI: kind=1 endpoint=2
  /// CHECK: Callback Fini:
