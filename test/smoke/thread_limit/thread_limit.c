#include <stdio.h>
#include <omp.h>

/*
 * Testing the passing of constant and variable arguments to thread_limit()
 */

int main()
{
  int N = 27;
  int NN = 1024;
  int varLimit[NN];
  int constLimit[NN];
  int thdLim = 27;
  int errors = 0;
  for (int i = 0; i < NN; i++)
   varLimit[i]=constLimit[i] -1;

  fprintf(stderr, "#pragma omp target teams distribute thread_limit(thdLim)\n");
#pragma omp target teams distribute thread_limit(thdLim)

  for (int i = 0; i < N; i++) {
    varLimit[i] = omp_get_num_threads();
  }
  for (int i = 0; i < N; i++) {
    fprintf(stderr, "varLimit[%d]: %d\n", i, varLimit[i]);
  }

  fprintf(stderr, "\n#pragma omp target teams distribute thread_limit(27)\n");
#pragma omp target teams distribute thread_limit(27)

  for (int i = 0; i < N; i++) {
    constLimit[i] = omp_get_num_threads();
  }
  for (int i = 0; i < N; i++) {
    fprintf(stderr, "constLimit[%d]: %d\n", i, constLimit[i]);
  }


 //Verify Results
 for (int i = 0; i < N; i++){
   if(varLimit[i] != constLimit[i] || constLimit[i] != 1){
	  errors++; 
   }
 }
  if(!errors)
    printf("Success\n");
  else
     printf("Fail\n");
  printf("Errors: %d\n", errors);
  return errors;

}

