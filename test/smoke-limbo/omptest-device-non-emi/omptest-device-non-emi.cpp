#include "omp.h"

#include "OmptTester.h"

using namespace omptest;
using namespace internal;

// Used in Test case later in the file
#define X 100000
#pragma omp declare target
int c[X];
#pragma omp end declare target

TEST(VeccopyCallbacks, OnDeviceDefaultTeams_Sequenced) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 100000;
  int i;

  int a[N];
  int b[N];

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 1);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);

#pragma omp target teams distribute parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    ;
  // EXPECT_EQ(a[i], b[i]);
}

TEST(VeccopyCallbacks, OnDeviceSetNumTeams_Sequenced) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 100000;
  int i;

  int a[N];
  int b[N];

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 1);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int)); // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 3);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);

#pragma omp target teams distribute parallel for num_teams(3)
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  int rc = 0;
  for (i = 0; i < N; i++)
    ;
  // EXPECT_EQ(a[i], b[i]);
}

TEST(VeccopyCallbacks, OnDevice_SequencedGrouped) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

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

TEST(VeccopyCallbacks, ExplicitDefaultDevice) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  int dev = omp_get_default_device();

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);

#pragma omp target teams distribute parallel for device(dev)
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  for (i = 0; i < N; i++)
    ; // EXPECT_EQ(a[i] == b[i]);
}

TEST(VeccopyCallbacks, MultipleDevices) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  for (int dev = 0; dev < omp_get_num_devices(); ++dev) {
    OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, dev);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int), a);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), a);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, N * sizeof(int), b);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, N * sizeof(int), b);
    OMPT_ASSERT_SEQUENCE(TargetSubmit, 0);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, b);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, N * sizeof(int), nullptr, a);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
    OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
    OMPT_ASSERT_SEQUENCE(Target, TARGET, END, dev);

#pragma omp target teams distribute parallel for device(dev)
    {
      for (int j = 0; j < N; j++)
        a[j] = b[j];
    }

    OMPT_ASSERT_SYNC_POINT("After target loop for device " +
                           std::to_string(dev));

    for (i = 0; i < N; i++)
      ; // EXPECT_EQ(a[i] == b[i]);
  } // device loop
}

TEST(DataOpTests, DeclareTargetGlobalAndUpdate) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int a[X];
  int b[X];

  int i;

  for (i = 0; i < X; i++)
    a[i] = 0;

  for (i = 0; i < X; i++)
    b[i] = i;

  for (i = 0; i < X; i++)
    c[i] = 0;

  OMPT_ASSERT_SEQUENCE(Target, ENTER_DATA, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, X * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, X * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(Target, ENTER_DATA, END, 0);

#pragma omp target enter data map(to : a)
  OMPT_ASSERT_SYNC_POINT("After enter data");

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, X * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, X * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 1);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, X * sizeof(int), nullptr, b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);
#pragma omp target parallel for
  {
    for (int j = 0; j < X; j++)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("After target parallel for");

  OMPT_ASSERT_SEQUENCE(Target, EXIT_DATA, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, X * sizeof(int), nullptr, a);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  OMPT_ASSERT_SEQUENCE(Target, EXIT_DATA, END, 0);
#pragma omp target exit data map(from : a)
  OMPT_ASSERT_SYNC_POINT("After target exit data");

  OMPT_ASSERT_SEQUENCE(Target, TARGET, BEGIN, 0);
  // FIXME: This is currently not occuring. Is this a bug?
  // OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, X * sizeof(c));
  OMPT_ASSERT_SEQUENCE(TargetSubmit, 1);
  OMPT_ASSERT_SEQUENCE(Target, TARGET, END, 0);
#pragma omp target parallel for map(alloc : c)
  {
    for (int j = 0; j < X; j++)
      c[j] = 2 * j + 1;
  }
  OMPT_ASSERT_SYNC_POINT("After target parallel for");

  OMPT_ASSERT_SEQUENCE(Target, UPDATE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, X * sizeof(int), nullptr, c);
  OMPT_ASSERT_SEQUENCE(Target, UPDATE, END, 0);
#pragma omp target update from(c) nowait
#pragma omp barrier

  for (i = 0; i < X; i++) {
    ;
    //    EXPECT_EQ (a[i] == i);
    //    EXPECT_EQ (c[i] == 2 * i + 1)
  }
}

TEST(DataOpTests, ExplicitAllocatorAPI) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *d_a = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  assert(d_a != nullptr);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  omp_target_free(d_a, omp_get_default_device());
}

TEST(DataOpTests, ExplicitAllocatorAndCopyAPI) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *d_a = nullptr;
  int *d_b = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, d_b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, d_b);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  d_b = (int *)omp_target_alloc(DataSize, omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, DataSize, d_a, d_b);
  omp_target_memcpy(d_b, d_a, DataSize, 0, 0, omp_get_default_device(),
                    omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0, d_b);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0, d_a);
  omp_target_free(d_b, omp_get_default_device());
  omp_target_free(d_a, omp_get_default_device());
}

TEST(DataOpTests, ExplicitAssociateDisassociatePtrAPI) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *p_a = nullptr;
  int *d_a = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);
  p_a = (int *)malloc(DataSize);
  memset(p_a, 0, N * sizeof(int));

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ASSOCIATE, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, omp_get_default_device());

  // DataSize unknown for disassociate in runtime
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DISASSOCIATE, 0, p_a);
  omp_target_disassociate_ptr(p_a, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  omp_target_free(d_a, omp_get_default_device());
  free(p_a);
}

TEST(DataOpTests, ExplicitAllocatorAndUpdate_NoDataOp) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *p_a = nullptr;
  int *d_a = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);
  p_a = (int *)malloc(DataSize);
  memset(p_a, 0, N * sizeof(int));

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ASSOCIATE, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(Target, UPDATE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(Target, UPDATE, END, 0);
  // No data operation: Compiler cannot deduce size
#pragma omp target update to(d_a)

  OMPT_ASSERT_SEQUENCE(TargetDataOp, DISASSOCIATE, 0, p_a);
  omp_target_disassociate_ptr(p_a, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  omp_target_free(d_a, omp_get_default_device());
  free(p_a);
}

TEST(DataOpTests, ExplicitAllocatorAndUpdate_DataOp) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *p_a = nullptr;
  int *d_a = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);
  p_a = (int *)malloc(DataSize);
  memset(p_a, 0, N * sizeof(int));

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ASSOCIATE, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(Target, UPDATE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, DataSize, p_a, d_a);
  OMPT_ASSERT_SEQUENCE(Target, UPDATE, END, 0);
#pragma omp target update to(p_a[0:N])

  OMPT_ASSERT_SEQUENCE(TargetDataOp, DISASSOCIATE, 0, p_a);
  omp_target_disassociate_ptr(p_a, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  omp_target_free(d_a, omp_get_default_device());
  free(p_a);
}

TEST(DataOpTests, ExplicitAllocatorAndUpdate_DataOpStack) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  const int N = 1000;
  int p_a[N];
  int *d_a = nullptr;
  auto DataSize = N * sizeof(int);
  memset(p_a, 0, N * sizeof(int));

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ASSOCIATE, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(Target, UPDATE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, H2D, DataSize, p_a, d_a);
  OMPT_ASSERT_SEQUENCE(Target, UPDATE, END, 0);
#pragma omp target update to(p_a)

  OMPT_ASSERT_SEQUENCE(TargetDataOp, DISASSOCIATE, 0, p_a);
  omp_target_disassociate_ptr(p_a, omp_get_default_device());
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0);
  omp_target_free(d_a, omp_get_default_device());
}

TEST_XFAIL(DataOpTests, ExplicitAllocatorMultiDevice) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *p_d1 = nullptr;
  int *p_d2 = nullptr;
  const int N = 100;
  auto DataSize = N * sizeof(int);

  auto dev1 = omp_get_initial_device();
  auto dev2 = (omp_get_initial_device() + 1) % omp_get_num_devices();

  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, p_d1);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, DataSize, p_d2);
  p_d1 = (int *)omp_target_alloc(DataSize, dev1);
  p_d2 = (int *)omp_target_alloc(DataSize, dev2);

  // According to specification, this should trigger an event.
  // However, given the current implementation, this test will fail
  // anyway and we need to look at it.
  // OMPT_ASSERT_SEQUENCE(TargetDataOp, MEMSET, DataSize, p_d1);
  omp_target_memset(p_d1, 0, DataSize, dev1);

  OMPT_ASSERT_SEQUENCE(TargetDataOp, D2H, DataSize, p_d1, p_d2);
  omp_target_memcpy(p_d2, p_d1, DataSize, 0, 0, dev2, dev1);

  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0, p_d1);
  OMPT_ASSERT_SEQUENCE(TargetDataOp, DELETE, 0, p_d2);
  omp_target_free(p_d1, dev1);
  omp_target_free(p_d2, dev2);
}

TEST(VeccopyTraces, OnHost_one) {
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);
  OMPT_PERMIT_EVENT(EventTy::ParallelBegin);
  OMPT_PERMIT_EVENT(EventTy::ThreadBegin);

  int N = 10;
  int a[N];
  int NumThreads = 2;

  OMPT_ASSERT_SET_NOT(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET_NOT(BufferRecord, CB_DATAOP, ALLOC, N * sizeof(int));

  OMPT_ASSERT_SET(ParallelBegin, NumThreads);
  OMPT_ASSERT_SET(ThreadBegin, ompt_thread_worker);

#pragma omp parallel for num_threads(NumThreads)
  {
    for (int i = 0; i < N; ++i) {
      a[i] = i;
    }
  }
}

TEST(VeccopyTraces, OnDeviceBufferRecord_parallel_for) {
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad);
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  OMPT_ASSERT_SET_MODE_STRICT();

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_KERNEL);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, END);
#pragma omp target parallel for map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("Final sync point");
}

TEST(VeccopyTraces, OnDeviceCallbacks_parallel_for) {
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  OMPT_ASSERT_SET_MODE_RELAXED();

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

  // This works with force-sync-regions, but not in default async execution.
  // Is this a bug in our implementation.
  OMPT_GENERATE_EVENTS(5, OMPT_ASSERT_SET(BufferRequest, 0, nullptr, 0));
  OMPT_GENERATE_EVENTS(
      11, OMPT_ASSERT_SET(BufferComplete, 0, nullptr, 0, 0, false));
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_KERNEL);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, END);
#pragma omp target parallel for map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("Final sync point");
}

TEST(VeccopyTraces, OnDeviceBufferRecord_teams_distribute_parallel_for) {
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_KERNEL);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, END);
#pragma omp target teams distribute parallel for map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("Final sync point");
}

TEST(VeccopyTraces, OnDeviceCallbacks_teams_distribute_parallel_for) {
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  OMPT_ASSERT_SET_MODE_RELAXED();

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

  // 2 data transfers TO device, 1 kernel, 2 data transfers FROM device
  OMPT_GENERATE_EVENTS(5, OMPT_ASSERT_SET(BufferRequest, 0, nullptr, 0));
  OMPT_GENERATE_EVENTS(
      11, OMPT_ASSERT_SET(BufferComplete, 0, nullptr, 0, 0, false));
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_KERNEL);
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, END);
#pragma omp target teams distribute parallel for map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
}

TEST(VeccopyTraces, OnDeviceBufferRecord_teams_distribute_parallel_for_nowait) {
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_KERNEL);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, END);
#pragma omp target teams distribute parallel for nowait map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("Final sync point");
}

TEST(VeccopyTraces, OnDeviceCallbacks_teams_distribute_parallel_for_nowait) {
  OMPT_SUPPRESS_EVENT(EventTy::Target);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  OMPT_ASSERT_SET_MODE_RELAXED();

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

  // 2 data transfers TO device, 1 kernel, 2 data transfers FROM device
  OMPT_GENERATE_EVENTS(5, OMPT_ASSERT_SET(BufferRequest, 0, nullptr, 0));
  OMPT_GENERATE_EVENTS(
      11, OMPT_ASSERT_SET(BufferComplete, 0, nullptr, 0, 0, false));
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, BEGIN);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_DATAOP);
  OMPT_ASSERT_SET(BufferRecord, CB_KERNEL);
  OMPT_ASSERT_SET(BufferRecord, CB_TARGET, TARGET, END);
#pragma omp target teams distribute parallel for nowait map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
}

// FIXME: Leave this suite here, so it gets discovered last and executed first.
TEST(InitFiniSuite, DeviceLoad) {
  OMPT_SUPPRESS_EVENT(EventTy::Target)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOp)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmit)
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 128;
  int a[N];

  for (int DeviceNum = 0; DeviceNum < omp_get_num_devices(); ++DeviceNum) {
    // Even on multi-GPU systems, only default-device is immediately initialized
    OMPT_ASSERT_SEQUENCE(DeviceLoad, /*DeviceNum=*/DeviceNum);

#pragma omp target parallel for device(DeviceNum)
    {
      for (int j = 0; j < N; j++)
        a[j] = 0;
    }

    OMPT_ASSERT_SYNC_POINT("After DeviceLoad " + std::to_string(DeviceNum));
  }
}

int main(int argc, char **argv) {
  Runner R;
  return R.run();
}
