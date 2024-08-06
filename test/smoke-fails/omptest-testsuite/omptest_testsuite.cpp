#include <stdio.h>

#include "OmptTester.h"

using namespace omptest;
using namespace internal;

TEST_XFAIL(SequenceSuite, uut_target_xfail_wrong_order) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit)

  int N = 128;
  int a[N];

  // Asserting the right observed events, but in wrong order.
  OMPT_ASSERT_SEQUENCE(Target, /*Kind=*/TARGET, /*Endpoint=*/END,
                       /*DeviceNum=*/0)
  OMPT_ASSERT_SEQUENCE(Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
                       /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST_XFAIL(SequenceSuite, uut_target_xfail_banned_begin_end) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit)

  int N = 128;
  int a[N];

  // Banning expected events should lead to a fail.
  OMPT_ASSERT_SEQUENCE_NOT(Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
                           /*DeviceNum=*/0)
  OMPT_ASSERT_SEQUENCE_NOT(Target, /*Kind=*/TARGET, /*Endpoint=*/END,
                           /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SequenceSuite, uut_target) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
                       /*DeviceNum=*/0)
  OMPT_ASSERT_SEQUENCE(Target, /*Kind=*/TARGET, /*Endpoint=*/END,
                       /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SequenceSuite, uut_target_dataop) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::Target)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(TargetDataOp, /*OpType=*/ALLOC, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE(TargetDataOp, /*OpType=*/H2D, /*Size=*/N * sizeof(int),
                       /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE(TargetDataOp, /*OpType=*/D2H, /*Size=*/N * sizeof(int),
                       /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE(TargetDataOp, /*OpType=*/DELETE, /*Size=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SequenceSuite, uut_target_submit) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::Target)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(TargetSubmit, /*RequestedNumTeams=*/1)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SequenceSuite, veccopy_ompt_target) {
  /* The Test Body */

  // Define testcase variables before events, so we can refer to them.
  int N = 100000;
  int a[N];
  int b[N];

  /* First Target Region */

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", Target, /*Kind=*/TARGET,
                               /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/ALLOC,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/H2D,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/ALLOC,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/H2D,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetSubmit, /*RequestedNumTeams=*/1)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/D2H,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/D2H,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/DELETE,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOp, /*OpType=*/DELETE,
                               /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", Target, /*Kind=*/TARGET, /*Endpoint=*/END,
                               /*DeviceNum=*/0)

  /* Second Target Region */

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", Target, /*Kind=*/TARGET,
                               /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/ALLOC,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/H2D,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/ALLOC,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/H2D,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetSubmit, /*RequestedNumTeams=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/D2H,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/D2H,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/DELETE,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOp, /*OpType=*/DELETE,
                               /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", Target, /*Kind=*/TARGET, /*Endpoint=*/END,
                               /*DeviceNum=*/0)

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

// Leave this suite down here so it gets discovered last and executed first.
TEST(InitialTestSuite, uut_device_init_load) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::Target)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(DeviceLoad, /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

int main(int argc, char **argv) {
  Runner R;
  return R.run();
}
