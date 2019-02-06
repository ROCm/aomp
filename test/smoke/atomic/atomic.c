#include <stdio.h>

#define N 100

int main()
{
  int fail = 0;
  int error = 0;
  int a = 0;
  int ii = 0;

  #pragma omp target map(tofrom:a)
  {
    #pragma omp parallel for
    for(ii = 0; ii < N; ++ii)
      #pragma omp atomic
      a++;
  }

  // Check result
  int result = a;
  int expect = N;
  if (result != expect)
  {
    printf("update (implicit) a %d != %d (error %d)\n",
        result, expect, ++error);
    fail = 1;
  }

  if(!fail)
    printf("successful\n");
  else
    fail = 0;

  printf("done with %d errors\n", error);
  return error;
}
