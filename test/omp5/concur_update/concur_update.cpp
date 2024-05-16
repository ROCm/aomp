#include <stdio.h>
#include <omp.h>
#include <stdint.h>

#define N 640
#define C 64
#define P 10

int A[N];

int main()
{
  // init
  int i;
  for(i=0; i<N; i++) A[i] = i;

  #pragma omp parallel num_threads(4)
  {
    int Num = omp_get_thread_num();
    fprintf(stderr, "Thread %d\n", Num);
    #pragma omp barrier
    #pragma omp target teams num_teams(4) map(tofrom: A)
    {
      for(int i=0; i<P; i++) {
        A[i] = i*5+Num;
      }
    }
  #pragma omp target update from( A)
  }

  int error = 0;
  for(i=0; i<P; i++) {
   fprintf(stderr, "%4d: got %d\n", i, A[i]);
  }
  fprintf(stderr,"completed TEST ARRAY\n");

  return 1; // it fails , its nondetermistic 
}
