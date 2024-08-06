#include <stdio.h>

#include "OmptTester.h"

TEST(ManualSuite, ParallelFor) {
  /* The Test Body */
  int arr[10] = {0};
  SequenceAsserter->insert(omptest::OmptAssertEvent::ParallelBegin(
      "User Parallel Begin", "Group", omptest::ObserveState::always,
      /*NumThreads=*/2));
  SequenceAsserter->insert(omptest::OmptAssertEvent::ThreadBegin(
      "User Thread Begin", "Group", omptest::ObserveState::always,
      ompt_thread_initial));
  SequenceAsserter->insert(omptest::OmptAssertEvent::ParallelEnd(
      "User Parallel End", "Group", omptest::ObserveState::always));

  SequenceAsserter->permitEvent(omptest::internal::EventTy::ParallelBegin);
  SequenceAsserter->permitEvent(omptest::internal::EventTy::ParallelEnd);
  SequenceAsserter->permitEvent(omptest::internal::EventTy::ThreadBegin);
  SequenceAsserter->permitEvent(omptest::internal::EventTy::ThreadEnd);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  SequenceAsserter->suppressEvent(omptest::internal::EventTy::ParallelBegin);
  SequenceAsserter->suppressEvent(omptest::internal::EventTy::ParallelEnd);
  SequenceAsserter->suppressEvent(omptest::internal::EventTy::ThreadBegin);
  SequenceAsserter->suppressEvent(omptest::internal::EventTy::ThreadEnd);
}

TEST(ManualSuite, ParallelForActivation) {
  /* The Test Body */
  int arr[10] = {0};
  SequenceAsserter->insert(omptest::OmptAssertEvent::ParallelBegin(
      "User Parallel Begin", "Group", omptest::ObserveState::always,
      /*NumThreads=*/2));
  SequenceAsserter->insert(omptest::OmptAssertEvent::ParallelEnd(
      "User Parallel End", "Group", omptest::ObserveState::always));

  // The last sequence we want to observe does not contain
  // another thread begin.
  SequenceAsserter->insert(omptest::OmptAssertEvent::ParallelBegin(
      "User Parallel Begin", "Group", omptest::ObserveState::always,
      /*NumThreads=*/2));
  SequenceAsserter->insert(omptest::OmptAssertEvent::ParallelEnd(
      "User Parallel End", "Group", omptest::ObserveState::always));

  SequenceAsserter->permitEvent(omptest::internal::EventTy::ParallelBegin);
  SequenceAsserter->permitEvent(omptest::internal::EventTy::ParallelEnd);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_ASSERT_SEQUENCE_DISABLE();

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_ASSERT_SEQUENCE_ENABLE();

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  SequenceAsserter->suppressEvent(omptest::internal::EventTy::ParallelBegin);
  SequenceAsserter->suppressEvent(omptest::internal::EventTy::ParallelEnd);
}

TEST(ManualSuite, AnotherTest) {
  int arr[12] = {0};

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 12; ++i) {
    arr[i] = i;
  }
}

TEST(ManualSuite, ParallelForToString) {
  /* The Test Body */
  int arr[10] = {0};

  OMPT_ASSERT_SEQUENCE_DISABLE();

  OMPT_REPORT_EVENT_DISABLE();

  EventReporter->permitEvent(omptest::internal::EventTy::ParallelBegin);
  EventReporter->permitEvent(omptest::internal::EventTy::ParallelEnd);
  EventReporter->permitEvent(omptest::internal::EventTy::ThreadBegin);
  EventReporter->permitEvent(omptest::internal::EventTy::ThreadEnd);

#pragma omp parallel for num_threads(2)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_REPORT_EVENT_ENABLE();

#pragma omp parallel for num_threads(3)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

#pragma omp parallel for num_threads(4)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

#pragma omp parallel for num_threads(5)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  EventReporter->suppressEvent(omptest::internal::EventTy::ParallelBegin);
  EventReporter->suppressEvent(omptest::internal::EventTy::ParallelEnd);
  EventReporter->suppressEvent(omptest::internal::EventTy::ThreadBegin);
  EventReporter->suppressEvent(omptest::internal::EventTy::ThreadEnd);

  OMPT_ASSERT_SEQUENCE_ENABLE();
}

TEST(ManualSuite, TargetParallelToString) {
  /* The Test Body */
  int arr[10] = {0};

  OMPT_ASSERT_SEQUENCE_DISABLE();
  OMPT_REPORT_EVENT_ENABLE();

#pragma omp target parallel for num_threads(1)
  for (int i = 0; i < 10; ++i)
    arr[i] = i;

  OMPT_ASSERT_SEQUENCE_ENABLE();
}

TEST(ManualSuite, veccopy_ompt_target) {
  /* The Test Body */
  OMPT_ASSERT_SEQUENCE_DISABLE();
  OMPT_REPORT_EVENT_ENABLE();

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

  OMPT_ASSERT_SEQUENCE_ENABLE();
}

int main(int argc, char **argv) {
  Runner R;
  return R.run();
}
