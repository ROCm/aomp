#include <stdio.h>

#include "OmptTester.h"

TEST(InitialSuite, veccopy) {
  int N = 10;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

  SequenceAsserter->insert(omptest::OmptAssertEvent::DeviceLoad(
      "User DeviceLoad", "Group", omptest::ObserveState::always,
      /*DeviceNum=*/0));

  SequenceAsserter->insert(omptest::OmptAssertEvent::Target(
      "User Target", "Group", omptest::ObserveState::always, /*Kind=*/TARGET,
      /*Endpoint=*/BEGIN));

  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/ALLOC, /*Bytes=*/sizeof(a)));
  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/H2D, /*Bytes=*/sizeof(a)));
  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/ALLOC, /*Bytes=*/sizeof(a)));
  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/H2D, /*Bytes=*/sizeof(a)));

  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetSubmit(
      "User Target Submit", "Group", omptest::ObserveState::always,
      /*RequestedNumTeams=*/1));

  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/D2H, /*Bytes=*/sizeof(a)));
  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/D2H, /*Bytes=*/sizeof(a)));
  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/DELETE));
  SequenceAsserter->insert(omptest::OmptAssertEvent::TargetDataOp(
      "User Target DataOp", "Group", omptest::ObserveState::always,
      /*OpType=*/DELETE));

  SequenceAsserter->insert(omptest::OmptAssertEvent::Target(
      "User Target", "Group", omptest::ObserveState::always, /*Kind=*/TARGET,
      /*Endpoint=*/END));

#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");
}

int main(int argc, char **argv) {
  Runner R;
  return R.run();
}
