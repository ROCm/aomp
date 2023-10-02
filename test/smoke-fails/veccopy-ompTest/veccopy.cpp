#include <stdio.h>

#include "OmptTester.h"

OMPTTESTCASE(InitialSuite, veccopy) {
  int N = 10;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

  
  SequenceAsserter.insert(omptest::OmptAssertEvent::Target("User Target"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetDataOp("User Target DataOp"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetDataOp("User Target DataOp"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetDataOp("User Target DataOp"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetSubmit("User Target Submit"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetDataOp("User Target DataOp"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetDataOp("User Target DataOp"));
  SequenceAsserter.insert(omptest::OmptAssertEvent::TargetDataOp("User Target DataOp"));


#pragma omp target parallel for
  {
    for (int j = 0; j< N; j++)
      a[j]=b[j];
  }

  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong varlue: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

}

int main(int argc, char **argv) {
  Runner R;
  R.run();

  return 0;
}
