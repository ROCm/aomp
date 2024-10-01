#include "omp.h"

#include "OmptTester.h"

using namespace omptest;
using namespace internal;

// Used in Test case later in the file
#define X 100000
#pragma omp declare target
int c[X];
#pragma omp end declare target

TEST(VeccopyCallbacks, OnDevice_Sequenced) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 100000;
  int i;

  int a[N];
  int b[N];

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 1, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 1, END);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                       &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                       &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, END, 0);

#pragma omp target parallel for
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // a ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // b ?
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), &b);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 3, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 3, END);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                       &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                       &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, END, 0);

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
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  // Define testcase variables before events, so we can refer to them.
  int N = 100000;
  int a[N];
  int b[N];

  /* First Target Region */

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC, BEGIN,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC, END,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC, BEGIN,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/ALLOC, END,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/H2D, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetSubmitEmi, /*RequestedNumTeams=*/1,
                               BEGIN)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetSubmitEmi, /*RequestedNumTeams=*/1,
                               END)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/D2H, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE, BEGIN,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE, END,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE, BEGIN,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetDataOpEmi, /*OpType=*/DELETE, END,
                               /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R1", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/END,
                               /*DeviceNum=*/0)

  /* Second Target Region */

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/BEGIN, /*DeviceNum=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC, BEGIN,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC, END,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC, BEGIN,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/ALLOC, END,
                               /*Size=*/N * sizeof(int))
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/H2D, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/&b)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetSubmitEmi, /*RequestedNumTeams=*/0,
                               BEGIN)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetSubmitEmi, /*RequestedNumTeams=*/0,
                               END)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&b)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H, BEGIN,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/D2H, END,
                               /*Size=*/N * sizeof(int),
                               /*SrcAddr=*/nullptr, /*DstAddr=*/&a)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE, BEGIN,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE, END,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE, BEGIN,
                               /*Size=*/0)
  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetDataOpEmi, /*OpType=*/DELETE, END,
                               /*Size=*/0)

  OMPT_ASSERT_SEQUENCE_GROUPED("R2", TargetEmi, /*Kind=*/TARGET,
                               /*Endpoint=*/END,
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
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
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

  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 0, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 0, END);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                       b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                       a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, END, 0);

#pragma omp target teams distribute parallel for device(dev)
  {
    for (int j = 0; j < N; j++)
      a[j] = b[j];
  }

  for (i = 0; i < N; i++)
    ; // EXPECT_EQ(a[i] == b[i]);
}

TEST(VeccopyCallbacks, MultipleDevices) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
    b[i] = i;

  for (int DeviceNum = 0; DeviceNum < omp_get_num_devices(); ++DeviceNum) {
    OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, BEGIN, DeviceNum);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int), a);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int), a);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), a);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), a);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int), b);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, N * sizeof(int), b);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), b);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, N * sizeof(int), b);
    OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 0, BEGIN);
    OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 0, END);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                         b);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr,
                         b);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr,
                         a);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr,
                         a);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
    OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
    OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, END, DeviceNum);

#pragma omp target teams distribute parallel for device(DeviceNum)
    {
      for (int j = 0; j < N; j++)
        a[j] = b[j];
    }

    OMPT_ASSERT_SYNC_POINT("After target loop for device " +
                           std::to_string(DeviceNum));

    for (i = 0; i < N; i++)
      ; // EXPECT_EQ(a[i] == b[i]);
  } // device loop
}

TEST(DataOpCallbacks, DeclareTargetGlobalAndUpdate) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
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

  OMPT_ASSERT_SEQUENCE(TargetEmi, ENTER_DATA, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, X * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, X * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, X * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, X * sizeof(int), a);
  OMPT_ASSERT_SEQUENCE(TargetEmi, ENTER_DATA, END, 0);

#pragma omp target enter data map(to : a)
  OMPT_ASSERT_SYNC_POINT("After enter data");

  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, X * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, X * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, X * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, X * sizeof(int), b);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 1, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 1, END);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, X * sizeof(int), nullptr,
                       b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, X * sizeof(int), nullptr, b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, END, 0);
#pragma omp target parallel for
  {
    for (int j = 0; j < X; j++)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("After target parallel for");

  OMPT_ASSERT_SEQUENCE(TargetEmi, EXIT_DATA, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, X * sizeof(int), nullptr,
                       a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, X * sizeof(int), nullptr, a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
  OMPT_ASSERT_SEQUENCE(TargetEmi, EXIT_DATA, END, 0);
#pragma omp target exit data map(from : a)
  OMPT_ASSERT_SYNC_POINT("After target exit data");

  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, BEGIN, 0);
  // FIXME: This is currently not occuring. Is this a bug?
  // OMPT_ASSERT_SEQUENCE(TargetDataOp, ALLOC, X * sizeof(c));
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 1, BEGIN);
  OMPT_ASSERT_SEQUENCE(TargetSubmitEmi, 1, END);
  OMPT_ASSERT_SEQUENCE(TargetEmi, TARGET, END, 0);
#pragma omp target parallel for map(alloc : c)
  {
    for (int j = 0; j < X; j++)
      c[j] = 2 * j + 1;
  }
  OMPT_ASSERT_SYNC_POINT("After target parallel for");

  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, X * sizeof(int), nullptr,
                       c);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, X * sizeof(int), nullptr, c);
  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, END, 0);
#pragma omp target update from(c) nowait
#pragma omp barrier

  for (i = 0; i < X; i++) {
    ;
    //    EXPECT_EQ (a[i] == i);
    //    EXPECT_EQ (c[i] == 2 * i + 1)
  }
}

TEST(DataOpCallbacks, ExplicitAllocatorAPI) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *d_a = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, DataSize);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, DataSize);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  assert(d_a != nullptr);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0);
  omp_target_free(d_a, omp_get_default_device());
}

TEST(DataOpCallbacks, ExplicitAllocatorAndCopyAPI) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *d_a = nullptr;
  int *d_b = nullptr;
  const int N = 1000;
  auto DataSize = N * sizeof(int);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, DataSize);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, DataSize, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, DataSize);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, DataSize, d_b);
  d_a = (int *)omp_target_alloc(DataSize, omp_get_default_device());
  d_b = (int *)omp_target_alloc(DataSize, omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, BEGIN, DataSize, d_a, d_b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, D2H, END, DataSize, d_a, d_b);
  omp_target_memcpy(d_b, d_a, DataSize, 0, 0, omp_get_default_device(),
                    omp_get_default_device());

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0, d_b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0, d_b);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0, d_a);
  omp_target_free(d_b, omp_get_default_device());
  omp_target_free(d_a, omp_get_default_device());
}

TEST(DataOpCallbacks, ExplicitAllocatorAndUpdate_NoDataOp) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *p_a = nullptr;
  int *d_a = nullptr;
  int DefaultDeviceNum = omp_get_default_device();
  const int N = 1000;
  auto DataSize = N * sizeof(int);
  p_a = (int *)malloc(DataSize);
  memset(p_a, 0, N * sizeof(int));

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, DataSize);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ASSOCIATE, BEGIN, DataSize, p_a, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ASSOCIATE, END, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, BEGIN, DefaultDeviceNum);
  // No data operation: Compiler cannot deduce size
  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, END, DefaultDeviceNum);
#pragma omp target update to(d_a)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DISASSOCIATE, BEGIN, 0, p_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DISASSOCIATE, END, 0, p_a);
  omp_target_disassociate_ptr(p_a, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0, d_a);
  omp_target_free(d_a, DefaultDeviceNum);
  free(p_a);
}

TEST(DataOpCallbacks, ExplicitAllocatorAndUpdate_DataOp) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  int *p_a = nullptr;
  int *d_a = nullptr;
  int DefaultDeviceNum = omp_get_default_device();
  const int N = 1000;
  auto DataSize = N * sizeof(int);
  p_a = (int *)malloc(DataSize);
  memset(p_a, 0, N * sizeof(int));

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, DataSize);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ASSOCIATE, BEGIN, DataSize, p_a, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ASSOCIATE, END, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, BEGIN, DefaultDeviceNum);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, DataSize, p_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, DataSize, p_a);
  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, END, DefaultDeviceNum);
#pragma omp target update to(p_a[0:N])

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DISASSOCIATE, BEGIN, 0, p_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DISASSOCIATE, END, 0, p_a);
  omp_target_disassociate_ptr(p_a, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0, d_a);
  omp_target_free(d_a, DefaultDeviceNum);
  free(p_a);
}

TEST(DataOpCallbacks, ExplicitAllocatorAndUpdate_DataOpStack) {
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecord);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRequest);
  OMPT_SUPPRESS_EVENT(EventTy::BufferComplete);
  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  const int N = 1000;
  int p_a[N];
  int *d_a = nullptr;
  int DefaultDeviceNum = omp_get_default_device();
  auto DataSize = N * sizeof(int);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, BEGIN, DataSize);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ALLOC, END, DataSize, d_a);
  d_a = (int *)omp_target_alloc(DataSize, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ASSOCIATE, BEGIN, DataSize, p_a, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, ASSOCIATE, END, DataSize, p_a, d_a);
  omp_target_associate_ptr(p_a, d_a, DataSize, 0, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, BEGIN, DefaultDeviceNum);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, BEGIN, DataSize, p_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, H2D, END, DataSize, p_a);
  OMPT_ASSERT_SEQUENCE(TargetEmi, UPDATE, END, DefaultDeviceNum);
#pragma omp target update to(p_a)

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DISASSOCIATE, BEGIN, 0, p_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DISASSOCIATE, END, 0, p_a);
  omp_target_disassociate_ptr(p_a, DefaultDeviceNum);

  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, BEGIN, 0, d_a);
  OMPT_ASSERT_SEQUENCE(TargetDataOpEmi, DELETE, END, 0, d_a);
  omp_target_free(d_a, DefaultDeviceNum);
}

TEST(VeccopyTraces, OnDeviceBufferRecord_parallel_for) {
  OMPT_SUPPRESS_EVENT(EventTy::DeviceLoad);
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi);
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
  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

  OMPT_ASSERT_SET_MODE_RELAXED();

  const int N = 10000;
  int a[N];
  int b[N];

  for (int i = 0; i < N; i++)
    a[i] = 0;

  for (int i = 0; i < N; i++)
    b[i] = i;

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
  OMPT_ASSERT_SET(TargetEmi, TARGET, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // a ?
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // a ?
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, END, N * sizeof(int), &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // b ?
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // b ?
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, END, N * sizeof(int), &b);
  OMPT_ASSERT_SET(TargetSubmitEmi, 1, BEGIN);
  OMPT_ASSERT_SET(TargetSubmitEmi, 1, END);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SET(TargetEmi, TARGET, END);
#pragma omp target parallel for map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
  OMPT_ASSERT_SYNC_POINT("Final sync point");
}

TEST(VeccopyTraces, OnDeviceBufferRecord_teams_distribute_parallel_for) {
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi);
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
  OMPT_ASSERT_SET(TargetEmi, TARGET, BEGIN, 0);
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // a ?
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // a ?
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, END, N * sizeof(int), &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // b ?
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // b ?
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, END, N * sizeof(int), &b);
  OMPT_ASSERT_SET(TargetSubmitEmi, 0, BEGIN);
  OMPT_ASSERT_SET(TargetSubmitEmi, 0, END);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SET(TargetEmi, TARGET, END, 0);
#pragma omp target teams distribute parallel for map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
}

TEST(VeccopyTraces, OnDeviceBufferRecord_teams_distribute_parallel_for_nowait) {
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi);
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi);
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi);
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
  OMPT_PERMIT_EVENT(EventTy::BufferRecord);

  OMPT_SUPPRESS_EVENT(EventTy::BufferRecordDeallocation);

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
  OMPT_ASSERT_SET(TargetEmi, TARGET, BEGIN, 0);
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // a ?
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // a ?
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, END, N * sizeof(int), &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, BEGIN, N * sizeof(int)); // b ?
  OMPT_ASSERT_SET(TargetDataOpEmi, ALLOC, END, N * sizeof(int));   // b ?
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, BEGIN, N * sizeof(int), &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, H2D, END, N * sizeof(int), &b);
  OMPT_ASSERT_SET(TargetSubmitEmi, 0, BEGIN);
  OMPT_ASSERT_SET(TargetSubmitEmi, 0, END);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &b);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, BEGIN, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, D2H, END, N * sizeof(int), nullptr, &a);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, BEGIN);
  OMPT_ASSERT_SET(TargetDataOpEmi, DELETE, END);
  OMPT_ASSERT_SET(TargetEmi, TARGET, END, 0);
#pragma omp target teams distribute parallel for nowait map(a, b)
  {
    for (int j = 0; j < N; ++j)
      a[j] = b[j];
  }
}

// FIXME: Leave this suite here, so it gets discovered last and executed first.
TEST(InitFiniSuite, DeviceLoad) {
  OMPT_SUPPRESS_EVENT(EventTy::TargetEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetDataOpEmi)
  OMPT_SUPPRESS_EVENT(EventTy::TargetSubmitEmi)
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
