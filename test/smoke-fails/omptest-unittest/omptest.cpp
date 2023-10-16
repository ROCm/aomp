#include <stdio.h>

#include "OmptTester.h"

OMPTTESTCASE(ManualSuite, ParallelFor) {
  /* The Test Body */
  int arr[10] = {0};
  SequenceAsserter.insert(omptest::OmptAssertEvent::ParallelBegin(
      /*NumThreads=*/2, "User Parallel Begin"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::ThreadBegin(
      ompt_thread_initial, "User Thread Begin"));
  SequenceAsserter.insert(
      omptest::OmptAssertEvent::ParallelEnd("User Parallel End"));

  SequenceAsserter.insert(omptest::OmptAssertEvent::ParallelEnd(""));

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;
}

OMPTTESTCASE(ManualSuite, ParallelForActivation) {
  /* The Test Body */
  int arr[10] = {0};
  SequenceAsserter.insert(omptest::OmptAssertEvent::ParallelBegin(
      /*NumThreads=*/2, "User Parallel Begin"));
  SequenceAsserter.insert(
      omptest::OmptAssertEvent::ParallelEnd("User Parallel End"));

  // The last sequence we want to observe does not contain
  // another thread begin.
  SequenceAsserter.insert(omptest::OmptAssertEvent::ParallelBegin(
      /*NumThreads=*/2, "User Parallel Begin"));
  SequenceAsserter.insert(
      omptest::OmptAssertEvent::ParallelEnd("User Parallel End"));

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_SEQ_ASSERT_DISABLE()

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_SEQ_ASSERT_ENABLE()

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;
}

OMPTTESTCASE(ManualSuite, AnotherTest) {
  int arr[12] = {0};

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 12; ++i) {
    arr[i] = i;
  }
}

OMPTTESTCASE(ManualSuite, ParallelForToString) {
  /* The Test Body */
  int arr[10] = {0};

  OMPT_SEQ_ASSERT_DISABLE()

  OMPT_EVENT_REPORT_DISABLE()

  EventReporter.permitEvent(omptest::internal::EventTy::ParallelBegin);
  EventReporter.permitEvent(omptest::internal::EventTy::ParallelEnd);
  EventReporter.permitEvent(omptest::internal::EventTy::ThreadBegin);
  EventReporter.permitEvent(omptest::internal::EventTy::ThreadEnd);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_EVENT_REPORT_ENABLE()

#pragma omp parallel for num_threads(3)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

#pragma omp parallel for num_threads(4)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

#pragma omp parallel for num_threads(5)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  EventReporter.suppressEvent(omptest::internal::EventTy::ParallelBegin);
  EventReporter.suppressEvent(omptest::internal::EventTy::ParallelEnd);
  EventReporter.suppressEvent(omptest::internal::EventTy::ThreadBegin);
  EventReporter.suppressEvent(omptest::internal::EventTy::ThreadEnd);

  OMPT_SEQ_ASSERT_ENABLE()
}

OMPTTESTCASE(ManualSuite, TargetParallelToString) {
  /* The Test Body */
  int arr[10] = {0};

  OMPT_SEQ_ASSERT_DISABLE()
  OMPT_EVENT_REPORT_ENABLE()

#pragma omp target parallel for num_threads(1)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_SEQ_ASSERT_ENABLE()
}

OMPTTESTCASE(ManualSuite, veccopy_ompt_target) {
  /* The Test Body */
  OMPT_SEQ_ASSERT_DISABLE()
  OMPT_EVENT_REPORT_ENABLE()

  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i = 0; i < N; i++)
    a[i] = 0;

  for (i = 0; i < N; i++)
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
  for (i = 0; i < N; i++)
    if (a[i] != b[i]) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  OMPT_SEQ_ASSERT_ENABLE()
}

int main(int argc, char **argv) {
  Runner R;
  R.run();

  return 0;
}
