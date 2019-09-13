#include <stdio.h>
#include <stdlib.h>
#define N 5000000

int main(){
  double *B,  *C;

  B = (double*)malloc(sizeof(double) * N);
  C = (double*)malloc(sizeof(double) * N);

  for(int i = 0; i < N; i++){
    B[i] = 1.0;
    C[i] = 1.0;
  }

  double sum = 0;
  
  #pragma omp target data map(to:B[0:N], C[0:N])
  #pragma omp target teams distribute parallel for map(tofrom:sum) reduction(+:sum)
  for(int i = 0; i < N; i++)
    sum += B[i] * C[i];

    printf("SUM = %f\n", sum);
  if (sum != N){
    printf("Failed!\n");
    return -1;
  } else{
    printf("SUCCESS!\n");
  }
  
  free(B);
  free(C);

  return 0;
}
