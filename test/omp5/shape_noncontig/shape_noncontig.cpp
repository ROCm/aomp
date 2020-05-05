#include <stdio.h>
#include <stdlib.h>
#define N 18

int main(){
  int *B,  *C;

  B = (int*)malloc(sizeof(int) * N);
  C = (int*)malloc(sizeof(int) * N);

  for(int i = 0; i < N; i++){
    C[i] = i;
    B[i] = 0;
  }

  int (&reshapeB)[3][6] = *reinterpret_cast<int(*)[3][6]>(B);
  int (&reshapeC)[3][6] = *reinterpret_cast<int(*)[3][6]>(C);
  int sum=0; 
#pragma omp target data map(tofrom:reshapeB[0:3][0:3:2], reshapeC[0:3][0:3:2])
//`#pragma omp target data map(tofrom:reshapeB[0:3][0:6], reshapeC[0:3][0:6])
  #pragma omp target teams distribute parallel for
  for(int i = 0; i < 3; i++)
    for(int j = 0; j < 3; j++)
      reshapeB[i][j] = reshapeC[i][j]*2;
  for(int i = 0; i < N; i++) {
      printf(" %d", B[i]);
      sum += B[i];
  }
  printf("\n");
  free(B);
  free(C);

   if (sum == 126) {
     printf("Passed\n");
     return 0;
   }
   printf("Failed\n");
   return 1;
}
