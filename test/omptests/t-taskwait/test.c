#include <stdlib.h>
#include <stdio.h>

#include "../utilities/check.h"

#define PRINT(_args...)
//#define PRINT(_args...) printf(_args)

#define N 64

#define TRY_TASK 1
#define TASK_COMPUTE 1

int main ()
{
  int a[N], aa[N];
  int b[N], bb[N];
  int c[N], cc[N];
  int d[N], dd[N];
  int e[N], ee[N];
  int i, errors;
  int cond = 0;

  check_offloading();

  // init
  for(i=0; i<N; i++) {
    a[i] = aa[i] = i+1;
    b[i] = bb[i] = 2*i +1;
  }

  // target starts 1 team and many threads in it
  #pragma omp target map(tofrom: a, b)
  {
    #pragma omp parallel num_threads(64)
    {
      int id = omp_get_thread_num();
      a[id]++;
      #pragma omp task firstprivate(id) shared(b)
      {
        PRINT("hi alex from  %d\n", id);
        int id = omp_get_thread_num();
        b[id]++;

        #pragma omp task firstprivate(id) shared(b)
        {
          PRINT("hi alex from  %d\n", id);
          int id = omp_get_thread_num();
          b[id]++;

          #pragma omp task firstprivate(id) shared(b)
          {
            PRINT("hi alex from  %d\n", id);
            int id = omp_get_thread_num();
            b[id]++;
          }
	  // wait for child to finish
	  #pragma omp taskwait
        }
	// wait for child to finish
        #pragma omp taskwait
      }
    }
  }

  // reproduce
  for(i=0; i<N; i++) {
    aa[i]++;
    bb[i]+=3;
  }

  // verify
  errors = 0;
  for(i=0; i<N; i++) {
    if (a[i] != aa[i]) printf("%4i: got a %d, expected %d, error %d\n", i, a[i], aa[i], ++errors);
    if (b[i] != bb[i]) printf("%4i: got b %d, expected %d, error %d\n", i, b[i], bb[i], ++errors);
  }
  printf("got %d errors\n", errors);

  return 0;
}
