// // From: https://github.com/OpenMP/Examples/blob/master/sources/Example_reduction.3.c
#include <stdio.h>

#define N   1000
#define SUM (N * (N-1)/2)

int main (void)
{
  int a, i;
  int error = 0;

  #pragma omp target parallel shared(a) private(i) map(tofrom: error)
  {
    #pragma omp master
    a = 0;

    #pragma omp barrier

    #pragma omp for reduction(+:a)
    for (i = 0; i < N; i++) {
        a += i;
    }

    // The Sum shall be sum:[0:N]
    #pragma omp single
    {
      if (a != SUM){
        printf ("Incorrect result = %d, expected = %d!\n", a, SUM);
	error = 1;
      }
      else{
        printf ("The result is correct = %d!\n", a);
	error = 0;
      }
    }
  }

  return error;
}
