#include <stdio.h>

#include "OmptTester.h"

using namespace omptest;
using namespace internal;

TEST(SequenceSuite, uut_target) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(TargetEmi, /*Kind=*/TARGET,
                       /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE(TargetEmi, /*Kind=*/TARGET,
                       /*Endpoint=*/END, /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SequenceSuite, uut_target_dataop) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/ALLOC,
                       /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/ALLOC,
                       /*Endpoint=*/END, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/H2D,
                       /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                       /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/H2D,
                       /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                       /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/D2H,
                       /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                       /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/D2H,
                       /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                       /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/DELETE,
                       /*Endpoint=*/BEGIN, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, /*OpType=*/DELETE,
                       /*Endpoint=*/END, /*Size=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SequenceSuite, uut_target_submit) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, /*RequestedNumTeams=*/1,
                       /*Endpoint=*/BEGIN)

  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, /*RequestedNumTeams=*/1,
                       /*Endpoint=*/END)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SetSuite, uut_target) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SET(TargetEmi, /*Kind=*/TARGET,
                  /*Endpoint=*/END, /*DeviceNum=*/0)

  OMPT_ASSERT_SET(TargetEmi, /*Kind=*/TARGET,
                  /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST_XFAIL(SetSuite, uut_target_xfail_banned_begin_end) {
  /* The Test Body */
  int N = 128;
  int a[N];

  // Banning expected events should lead to a fail.
  OMPT_ASSERT_SET_NOT(TargetEmi, /*Kind=*/TARGET, /*Endpoint=*/END,
                      /*DeviceNum=*/0)
  OMPT_ASSERT_SET_NOT(TargetEmi, /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
                      /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SetSuite, uut_target_ignore_non_emi_events) {
  /* The Test Body */
  int N = 128;
  int a[N];

  // Ignore other additional events
  OMPT_ASSERT_SET_MODE_RELAXED()

  // We should not observe non-EMI events
  OMPT_ASSERT_SET_NOT(Target, /*Kind=*/TARGET, /*Endpoint=*/END,
                      /*DeviceNum=*/0)
  OMPT_ASSERT_SET_NOT(Target, /*Kind=*/TARGET, /*Endpoint=*/BEGIN,
                      /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SetSuite, uut_target_dataop) {
  /* The Test Body */
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/DELETE,
                  /*Endpoint=*/END, /*Size=*/0)

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/D2H,
                  /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/ALLOC,
                  /*Endpoint=*/END, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/H2D,
                  /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&a)

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/H2D,
                  /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/&a)

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/D2H,
                  /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                  /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/ALLOC,
                  /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SET(TargetDataOpEmi, /*OpType=*/DELETE,
                  /*Endpoint=*/BEGIN, /*Size=*/0)

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

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetSubmitEmi, /*RequestedNumTeams=*/1,
                               /*Endpoint=*/BEGIN)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetSubmitEmi, /*RequestedNumTeams=*/1,
                               /*Endpoint=*/END)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/BEGIN, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/END, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/BEGIN, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/END, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/END, /*DeviceNum=*/0)

  /* Second Target Region */

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int))

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetSubmitEmi, /*RequestedNumTeams=*/0,
                               /*Endpoint=*/BEGIN)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetSubmitEmi, /*RequestedNumTeams=*/0,
                               /*Endpoint=*/END)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/BEGIN, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H,
                               /*Endpoint=*/END, /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/BEGIN, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/END, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/BEGIN, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE,
                               /*Endpoint=*/END, /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/END, /*DeviceNum=*/0)

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
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)

  int N = 128;
  int a[N];

  OMPT_ASSERT_SET(DeviceLoad, /*DeviceNum=*/0)

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
