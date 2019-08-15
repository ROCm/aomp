#include <stdio.h>

#define N   1000000ll
#define SUM (N * (N-1)/2)

int main (void)
{

  #pragma omp target
  {
    long long a, i;
    a = 0;

    #pragma omp parallel for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
    }
    {
      if (a != SUM)
        printf ("Incorrect result = %lld, expected = %lld!\n", a, SUM);
      else
        printf ("The result is correct = %lld!\n", a);
    }
  }

  return 0;
}

