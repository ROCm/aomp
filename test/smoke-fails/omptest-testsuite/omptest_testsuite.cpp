#include <stdio.h>

#include "OmptTester.h"

using namespace omptest;
using namespace internal;

OMPTTESTCASE(ManualSuite, veccopy_ompt_target) {
  /* The Test Body */

  // Define testcase variables before events, so we can refer to them.
  int N = 100000;
  int a[N];
  int b[N];

  /* First Target Region */

  OMPT_SEQ_ASSERT(DeviceInitialize, /*DeviceNum=*/0);
  OMPT_SEQ_ASSERT(DeviceLoad, /*DeviceNum=*/0);

  OMPT_SEQ_ASSERT_GROUP("R1", Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/0);

  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/ALLOC, /*Size=*/N * sizeof(int));
  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/H2D, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&a);

  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/ALLOC, /*Size=*/N * sizeof(int));
  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/H2D, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&b);

  OMPT_SEQ_ASSERT_GROUP("R1", TargetSubmit, /*RequestedNumTeams=*/1);

  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/D2H, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&b);
  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/D2H, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&a);

  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/DELETE, /*Size=*/0);
  OMPT_SEQ_ASSERT_GROUP("R1", TargetDataOp, /*OpType=*/DELETE, /*Size=*/0);

  OMPT_SEQ_ASSERT_GROUP("R1", Target, /*Kind=*/TARGET, /*Endpoint=*/END, /*DeviceNum=*/0);

  /* Second Target Region */

  OMPT_SEQ_ASSERT_GROUP("R2", Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/0);

  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/ALLOC,/*Size=*/N * sizeof(int));
  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/H2D, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&a);

  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/ALLOC, /*Size=*/N * sizeof(int));
  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/H2D, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&b);

  OMPT_SEQ_ASSERT_GROUP("R2", TargetSubmit, /*RequestedNumTeams=*/0);

  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/D2H, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&b);
  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/D2H, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&a);

  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/DELETE, /*Size=*/0);
  OMPT_SEQ_ASSERT_GROUP("R2", TargetDataOp, /*OpType=*/DELETE, /*Size=*/0);

  OMPT_SEQ_ASSERT_GROUP("R2", Target, /*Kind=*/TARGET, /*Endpoint=*/END, /*DeviceNum=*/0);

  /* Actual testcase code. */

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
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
  for (int i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");
}

OMPTTESTCASE(ManualSuite, uut_target) {
  /* The Test Body */
  OMPT_EVENT_SUPPRESS_EACH_LISTENER(EventTy::TargetDataOp);
  OMPT_EVENT_SUPPRESS_EACH_LISTENER(EventTy::TargetSubmit);

  int N = 128;
  int a[N];

  OMPT_SEQ_ASSERT(Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/0);
  OMPT_SEQ_ASSERT(Target, /*Kind=*/TARGET, /*Endpoint=*/END, /*DeviceNum=*/0);

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }

  OMPT_SEQ_ASSERT_DISABLE()
}

OMPTTESTCASE(ManualSuite, uut_target_dataop) {
  /* The Test Body */
  OMPT_EVENT_SUPPRESS_EACH_LISTENER(EventTy::Target);
  OMPT_EVENT_SUPPRESS_EACH_LISTENER(EventTy::TargetSubmit);

  int N = 128;
  int a[N];

  OMPT_SEQ_ASSERT(TargetDataOp, /*OpType=*/ALLOC, /*Size=*/N * sizeof(int));

  OMPT_SEQ_ASSERT(TargetDataOp, /*OpType=*/H2D, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&a);

  OMPT_SEQ_ASSERT(TargetDataOp, /*OpType=*/D2H, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&a);

  OMPT_SEQ_ASSERT(TargetDataOp, /*OpType=*/DELETE, /*Size=*/0);

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

OMPTTESTCASE(ManualSuite, uut_target_submit) {
  /* The Test Body */
  OMPT_EVENT_SUPPRESS_EACH_LISTENER(EventTy::Target);
  OMPT_EVENT_SUPPRESS_EACH_LISTENER(EventTy::TargetDataOp);

  int N = 128;
  int a[N];

  OMPT_SEQ_ASSERT(TargetSubmit, /*RequestedNumTeams=*/1);

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

int main(int argc, char **argv) {
  libomptest_global_eventreporter_set_active(false);
  Runner R;
  R.run();

  return 0;
}
