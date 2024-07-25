#include "OmptTester.h"

#include <omp.h>

#ifdef __cplusplus
extern "C" {
#endif
ompt_set_result_t libomptarget_ompt_set_trace_ompt(int DeviceId,
                                                   unsigned int Enable,
                                                   unsigned int EventTy);
#ifdef __cplusplus
}
#endif

using namespace omptest;
using namespace internal;

TEST(MultiGPUTracingSuite, uut_device_set_trace_disabled) {
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  int HostId = omp_get_initial_device();
  int NumDevices = omp_get_num_devices();

  // We should not observe any trace records, on any device
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_TARGET,
                      /*Kind=*/TARGET, /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_TARGET,
                      /*Kind=*/TARGET, /*Endpoint=*/END)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/ALLOC)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/H2D)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/D2H)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/DELETE)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_KERNEL)

  for (int i = 0; i < NumDevices; ++i) {
    std::string SyncPointName{"Device " + std::to_string(i) +
                              ": Disabled tracing"};

    // Disable tracing on device for all event types
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/0,
                                     /*EventyTy=*/0);

#pragma omp target parallel for device(i)
    {
      for (int j = 0; j < N; j++)
        a[j] = 0;
    }

    // We should not observe any events
    OMPT_ASSERT_SYNC_POINT(SyncPointName)

    // Finally: re-enable tracing for all event types
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/0);
  }
}

TEST(MultiGPUTracingSuite, uut_device_set_trace_target) {
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  int HostId = omp_get_initial_device();
  int NumDevices = omp_get_num_devices();

  // We should only observe target region trace records
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/ALLOC)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/H2D)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/D2H)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/DELETE)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_KERNEL)

  for (int i = 0; i < NumDevices; ++i) {
    std::string SyncPointName{"Device " + std::to_string(i) +
                              ": Enabled target tracing"};

    // Enable tracing on device only for data operations
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/0,
                                     /*EventyTy=*/0);
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/CB_TARGET);

    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_TARGET,
                    /*Kind=*/TARGET, /*Endpoint=*/BEGIN, /*DeviceNum=*/i)
    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_TARGET,
                    /*Kind=*/TARGET, /*Endpoint=*/END, /*DeviceNum=*/i)

#pragma omp target parallel for device(i)
    {
      for (int j = 0; j < N; j++)
        a[j] = 0;
    }
    OMPT_ASSERT_SYNC_POINT(SyncPointName)

    // Finally: re-enable tracing for all event types
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/0);
  }
}

TEST(MultiGPUTracingSuite, uut_device_set_trace_data_op) {
  int N = 128;
  int a[N];

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  int HostId = omp_get_initial_device();
  int NumDevices = omp_get_num_devices();

  // We should only observe data operation trace records
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_TARGET,
                      /*Kind=*/TARGET, /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_TARGET,
                      /*Kind=*/TARGET, /*Endpoint=*/END)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_KERNEL)

  for (int i = 0; i < NumDevices; ++i) {
    std::string SyncPointName{"Device " + std::to_string(i) +
                              ": Enabled data op tracing"};

    // Enable tracing on device only for data operations
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/0,
                                     /*EventyTy=*/0);
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/CB_DATAOP);

    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP,
                    /*OpType=*/ALLOC, /*Bytes=*/sizeof(a),
                    /*MinimumTimeDelta=*/10, /*SrcAddr=*/nullptr,
                    /*DstAddr=*/nullptr,
                    /*SrcDeviceNum=*/HostId, /*DstDeviceNum=*/i)
    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP,
                    /*OpType=*/H2D, /*Bytes=*/sizeof(a),
                    /*MinimumTimeDelta=*/10, /*SrcAddr=*/nullptr,
                    /*DstAddr=*/nullptr,
                    /*SrcDeviceNum=*/HostId, /*DstDeviceNum=*/i)
    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP,
                    /*OpType=*/D2H, /*Bytes=*/sizeof(a),
                    /*MinimumTimeDelta=*/10, /*SrcAddr=*/nullptr,
                    /*DstAddr=*/nullptr,
                    /*SrcDeviceNum=*/i, /*DstDeviceNum=*/HostId)
    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_DATAOP,
                    /*OpType=*/DELETE, /*Bytes=*/0,
                    /*MinimumTimeDelta=*/0, /*SrcAddr=*/nullptr,
                    /*DstAddr=*/nullptr,
                    /*SrcDeviceNum=*/i, /*DstDeviceNum=*/-1)

#pragma omp target parallel for device(i)
    {
      for (int j = 0; j < N; j++)
        a[j] = 0;
    }
    OMPT_ASSERT_SYNC_POINT(SyncPointName)

    // Finally: re-enable tracing for all event types
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/0);
  }
}

TEST(MultiGPUTracingSuite, uut_device_set_trace_target_submit) {
  int N = 128;
  int a[N];

  // FIXME This test is failing (if more than one device is present) due to a
  // missing kernel trace record from the 'last' DeviceId.
  // Note: This is fixed if a second (identical) kernel is asserted and started.
  // Note: Running with each single GPU using ROCR_VISIBLE_DEVICES does pass.
  // Note: The device order changes the behavior, e.g. with 2 GPUs:
  //       for (int i = 0; i < NumDevices; ++i)
  //       [0,1]: Id=0 will pass, Id=1 will fail (no trace record)
  //       for (int i = (NumDevices - 1); i >= 0; --i)
  //       [1,0]: Id=1 will fail, Id=0 will pass (but trace record is from Id=0)

  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest)
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete)

  int HostId = omp_get_initial_device();
  int NumDevices = omp_get_num_devices();

  // We should only observe target submit / kernel trace records
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_TARGET,
                      /*Kind=*/TARGET, /*Endpoint=*/BEGIN)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_TARGET,
                      /*Kind=*/TARGET, /*Endpoint=*/END)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/ALLOC)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/H2D)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/D2H)
  OMPT_ASSERT_SET_NOT(BufferRecord, /*Type=*/CB_DATAOP, /*OpType=*/DELETE)

  for (int i = 0; i < NumDevices; ++i) {
    std::string SyncPointName{"Device " + std::to_string(i) +
                              ": Enabled kernel tracing"};

    // Enable tracing on device only for kernels
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/0,
                                     /*EventyTy=*/0);
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/CB_KERNEL);

    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_KERNEL)

    // FIXME We should not need a second kernel launch
    OMPT_ASSERT_SET(BufferRecord, /*Type=*/CB_KERNEL)

#pragma omp target parallel for device(i)
    {
      for (int j = 0; j < N; j++)
        a[j] = 0;
    }

    // FIXME We should not need a second kernel launch
#pragma omp target parallel for device(i)
    {
      for (int j = 0; j < N; j++)
        a[j] = 0;
    }

    OMPT_ASSERT_SYNC_POINT(SyncPointName)

    // Finally: re-enable tracing for all event types
    libomptarget_ompt_set_trace_ompt(/*DeviceId=*/i, /*Enable=*/1,
                                     /*EventyTy=*/0);
  }
}

TEST(DisabledSuite, DISABLED_uut_target_disabled) {
  assert(false && "This test should have been disabled");
}

TEST(SequenceAssertion, uut_target_sequence) {
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
TEST(InitialTestSuite, uut_device_init_load_multi_gpu) {
  // We only want to assert on DeviceLoads: ignore other events
  OMPT_ASSERT_SET_MODE_RELAXED()

  for (int i = 0; i < omp_get_num_devices(); ++i) {
    OMPT_ASSERT_SET(DeviceLoad, /*DeviceNum=*/i)
#pragma omp target device(i)
    {
      ;
    }
  }
}

#ifndef LIBOFFLOAD_LIBOMPTEST_USE_GOOGLETEST
int main(int argc, char **argv) {
  Runner R;
  return R.run();
}
#endif
