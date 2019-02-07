
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  int dep = 0;

  #pragma omp target device(0) nowait map(tofrom: dep) depend(out: dep)
  {
    dep++;
  }

  #pragma omp target device(1) nowait map(tofrom: dep) depend(in: dep)
  {
    dep++;
  }
  #pragma omp taskwait

  if (dep == 2) {
    printf("completed with 0 errors\n");
  } else {
    printf("completed with a error:\n");
    printf("dep should be 2, but is %d\n", dep);
  }

  return EXIT_SUCCESS;
}
