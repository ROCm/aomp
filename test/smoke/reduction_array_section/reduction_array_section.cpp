#include <iostream>
#include <omp.h>
#define SIZE 6
#define VAL 1
int main()
{
  int outer = 50;
  int errors = 0;
  int myArray[SIZE] = {};
  #pragma omp target data map(tofrom:myArray[0:SIZE])
  #pragma omp target parallel for reduction(+:myArray[:SIZE])
  for (int i=0; i < outer; i++)
  {
    int a = VAL; 
    for (int n = 0; n < SIZE; n++)
    {
      myArray[n] += a;
    }
  }

  for (int i = 0; i < SIZE; i++)
  {
    printf("myArray[%d]: %d\n", i, myArray[i]);
    if(SIZE > 1 && myArray[i] != outer * VAL){
      printf("Error! myArray[%d] is %d and should be %d\n", i, myArray[i], outer * VAL);
      errors++;
    }
    if(SIZE == 1 && myArray[0] != outer * SIZE){
      printf("Error! myArray[0] is %d and should be %d\n", myArray[0], outer * SIZE);
      errors++;
    }

  } 
  if(errors){
    printf("Fail!\n");
    return 1;
  }
    printf("Success!\n");
    return 0;
}
