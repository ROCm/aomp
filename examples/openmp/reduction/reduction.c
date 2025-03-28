#include <stdio.h>
#define N   1000000ll
#define SUM (N * (N-1)/2)
int main (void) {
  long long a, i;
  a = 0;
  #pragma omp target parallel for reduction(+:a)
  for (i = 0; i < N; i++)
    a += i;

  if (a != SUM) {
    printf("Incorrect result = %lld, expected = %lld!\nFail!\n", a, SUM);
    return 1;
  }
  printf ("The result is correct %lld!\nSuccess!\n", a);
  return 0;
}
