#include <stdio.h>
#include <stdlib.h>

#define N 1000

class A0 {
public:
  int a, b, c, sum, *p;
  A0() {
    a = 1; b = 2; c = 3;
    p = (int *)malloc(N*sizeof(int));
    for(int i=0; i<N; i++) p[i] = 10+i;
  }
  ~A0() {
    free(p);
  }

  int Num() {
    #pragma omp target 
    {
      sum = a + b + c;
    }
    return sum;
  }
};

int main() {

  A0 a0;
  if (a0.Num() != (1+2+3)) {
    printf("Fail\n");
    return 1;
  } else {
    printf("Success\n");
    return 0;
  }

  return 0;
}
