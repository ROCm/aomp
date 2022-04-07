struct test_type {
    int *p1;
} tt;
#include "stdlib.h"
#include "stdio.h"
int C[10];
int E[10];
int main()
{
    int i;
    int *p;
      tt.p1 = (int*) malloc(10*sizeof(int));
      p=tt.p1;
      for (i=0; i<10; i++) {
          tt.p1[i] = i+100;
          C[i] = 0;
      }
      for (i=0; i<10; i++)
        E[i]=C[i]+10 + tt.p1[i];
      #pragma omp target map(tofrom: C) map(to: tt, tt.p1[:10])
      {
        for (i=0; i<10; i++)
          C[i]=C[i]+10 + tt.p1[i];
      }
      for (i=0; i<10; i++) {
         printf("%d \n", C[i]);
	 if (E[i] != C[i]) return 1;
      }
      return 0;
}



