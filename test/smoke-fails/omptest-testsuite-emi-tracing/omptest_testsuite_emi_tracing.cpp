#include "OmptTester.h"

using namespace omptest;
using namespace internal;

TEST(DisabledSuite, DISABLED_uut_target_disabled) {
  assert(false && "This test should have been disabled");
}

TEST(SequenceAssertion, uut_target_sequence) {
  /* The Test Body */
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_TARGET, /*Kind=*/TARGET,
                       /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP,
                       /*OpType=*/ALLOC, /*Bytes=*/sizeof(a),
                       /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP,
                       /*OpType=*/H2D, /*Bytes=*/sizeof(a),
                       /*Timeframe=*/{10, 100000})

  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_KERNEL)

  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/D2H)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/DELETE)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_TARGET, /*Kind=*/TARGET,
                       /*Endpoint=*/END)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }

  OMPT_ASSERT_SYNC_POINT("S1")
}

TEST(SetAssertion, uut_target_set) {
  /* The Test Body */
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_TARGET, /*Kind=*/TARGET,
                  /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP,
                  /*OpType=*/ALLOC, /*Bytes=*/sizeof(a),
                  /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP,
                  /*OpType=*/H2D, /*Bytes=*/sizeof(a),
                  /*Timeframe=*/{10, 100000})

  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_KERNEL, /*MinimumTimeDelta=*/100)

  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/D2H,
                  /*Bytes=*/sizeof(a), /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/DELETE,
                  /*Bytes=*/0, /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_TARGET, /*Kind=*/TARGET,
                  /*Endpoint=*/END)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(SetAssertion, uut_target_set_generic_duplicates) {
  /* The Test Body */
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  OMPT_GENERATE_EVENTS(2, OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_TARGET))
  OMPT_GENERATE_EVENTS(4, OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP))
  OMPT_GENERATE_EVENTS(1, OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_KERNEL))

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(AssertionMode, uut_target_set_mode_relaxed) {
  /* The Test Body */
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  OMPT_ASSERT_SET_MODE_RELAXED()

  OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_KERNEL)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(AssertionMode, uut_target_sequence_mode_relaxed) {
  /* The Test Body */
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  OMPT_ASSERT_SEQUENCE_MODE_RELAXED()

  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_TARGET)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_KERNEL)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_DATAOP)
  OMPT_ASSERT_SEQUENCE(BufferRecord, /*Type=*/CB_TARGET)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

TEST(MixedAssertionSuite, uut_target_sequence_and_set) {
  /* The Test Body */
  int N = 128;
  int a[N];
  int b[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  // This will create / delete groups using sequential asserter
  OMPT_ASSERT_SEQUENCE_GROUPED_ONLY("R1", TargetEmi, /*Kind=*/TARGET,
                             /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SEQUENCE_GROUPED_ONLY("R1", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/END)

  // Set asserter may also create all groups in parallel
  OMPT_ASSERT_SET_GROUPED("R1", TargetEmi, /*Kind=*/TARGET,
                          /*Endpoint=*/BEGIN)

  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_TARGET,
                          /*Kind=*/TARGET,
                          /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/ALLOC, /*Bytes=*/sizeof(a),
                          /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/H2D, /*Bytes=*/sizeof(a),
                          /*Timeframe=*/{10, 100000})
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/ALLOC, /*Bytes=*/sizeof(b),
                          /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/H2D, /*Bytes=*/sizeof(b),
                          /*Timeframe=*/{10, 100000})

  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_KERNEL)

  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/D2H)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/D2H)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/DELETE)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/DELETE)
  OMPT_ASSERT_SET_GROUPED("R1", BufferRecord, /*Type=*/CB_TARGET,
                          /*Kind=*/TARGET, /*Endpoint=*/END)

  OMPT_ASSERT_SET_GROUPED("R1", TargetEmi, /*Kind=*/TARGET,
                          /*Endpoint=*/END)

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

  // Ensure there are no remaining expected events
  OMPT_ASSERT_SYNC_POINT("S1")

  OMPT_ASSERT_SET_GROUPED("R2", TargetEmi, /*Kind=*/TARGET,
                          /*Endpoint=*/BEGIN)

  OMPT_ASSERT_SEQUENCE_GROUPED_ONLY("R2", TargetEmi, /*Kind=*/TARGET,
                                    /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SEQUENCE_GROUPED_ONLY("R2", TargetEmi, /*Kind=*/TARGET,
                                    /*Endpoint=*/END)

  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_TARGET,
                          /*Kind=*/TARGET,
                          /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/ALLOC, /*Bytes=*/sizeof(a),
                          /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/H2D, /*Bytes=*/sizeof(a),
                          /*Timeframe=*/{10, 100000})
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/ALLOC, /*Bytes=*/sizeof(b),
                          /*MinimumTimeDelta=*/10)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/H2D, /*Bytes=*/sizeof(b),
                          /*Timeframe=*/{10, 100000})

  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_KERNEL)

  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/D2H)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/D2H)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/DELETE)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_DATAOP,
                          /*OpType=*/DELETE)
  OMPT_ASSERT_SET_GROUPED("R2", BufferRecord, /*Type=*/CB_TARGET,
                          /*Kind=*/TARGET, /*Endpoint=*/END)

  OMPT_ASSERT_SET_GROUPED("R2", TargetEmi, /*Kind=*/TARGET,
                          /*Endpoint=*/END)

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  OMPT_ASSERT_SYNC_POINT("S2")

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
  int N = 128;
  int a[N];

  OMPT_ASSERT_SET(DeviceLoad, /*DeviceNum=*/0)

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = 0;
  }
}

#ifndef LIBOFFLOAD_LIBOMPTEST_USE_GOOGLETEST
int main(int argc, char **argv) {
  Runner R;
  R.run();

  return 0;
}
#endif
