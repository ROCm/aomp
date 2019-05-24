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
  int varLimitHuge[NN];
  int constLimit[NN];
  int constLimitHuge[NN];
  int thdLim = 27;
  int errors = 0;
  for (int i = 0; i < NN; i++)
   varLimit[i]=constLimit[i]=constLimitHuge[i]=varLimitHuge[i] = -1;

  fprintf(stderr, "#pragma omp target teams distribute thread_limit(thdLim) 27\n");
#pragma omp target teams distribute thread_limit(thdLim)

  for (int i = 0; i < N; i++) {
    varLimit[i] = omp_get_num_threads();
  }
  for (int i = 0; i < N; i++) {
    fprintf(stderr, "varLimit[%d]: %d\n", i, varLimit[i]);
  }

  thdLim = 1024;
  fprintf(stderr, "#pragma omp target teams distribute thread_limit(thdLim) 1024\n");
#pragma omp target teams distribute thread_limit(thdLim)

  for (int i = 0; i < N; i++) {
    varLimitHuge[i] = omp_get_num_threads();
  }
  for (int i = 0; i < N; i++) {
    fprintf(stderr, "varLimitHuge[%d]: %d\n", i, varLimitHuge[i]);
  }

  fprintf(stderr, "\n#pragma omp target teams distribute thread_limit(27)\n");
#pragma omp target teams distribute thread_limit(27)

  for (int i = 0; i < N; i++) {
    constLimit[i] = omp_get_num_threads();
  }
  for (int i = 0; i < N; i++) {
    fprintf(stderr, "constLimit[%d]: %d\n", i, constLimit[i]);
  }

  fprintf(stderr, "\n#pragma omp target teams distribute thread_limit(1024)\n");
#pragma omp target teams distribute thread_limit(1024)

  for (int i = 0; i < N; i++) {
    constLimitHuge[i] = omp_get_num_threads();
  }
  for (int i = 0; i < N; i++) {
    fprintf(stderr, "constLimitHuge[%d]: %d\n", i, constLimitHuge[i]);
  }


 //Verify Results
 for (int i = 0; i < N; i++){
   if(varLimit[i] != constLimit[i] || constLimit[i] != 1 ||
      varLimitHuge[i] != constLimit[i] || varLimitHuge[i] != 1 ||
      varLimit[i] != constLimitHuge[i] || constLimitHuge[i] != 1){
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

