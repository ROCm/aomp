#include <stdio.h>

int main(void) {
  int A[1] = {0};
  
  try
  {
    #pragma omp target
    ++A[0];
    
    throw 123;
  }
  catch (int e)
  {
    #pragma omp target
    {
      int a = 0;
      ++a;
      #pragma omp parallel num_threads(1)
        A[0] += a;
    }
    printf("Exception %d\n", 123);
  }

  #pragma omp target
  ++A[0];

  printf("Got %d, should be 3\n", A[0]);
  return 0;
}
