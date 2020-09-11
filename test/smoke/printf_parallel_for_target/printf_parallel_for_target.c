#include <omp.h>
#include <stdio.h>
int main() {
  int verbose = 0;
  if(getenv("VERBOSE_PRINT")){
    // Note: verbose mode does not verify output as printf thread output and addresses will differ each run.
    verbose = 1;
    printf("Using verbose print...printing pointer addresses.\n");
  }
    
  #pragma omp parallel for                                                                                                                                                
  for (int j = 0; j < 2; j++) {
    if(verbose)
      printf("CPU Parallel: Hello from %d\n", omp_get_thread_num());
    else
      printf("CPU Parallel: Hello.\n");
  #pragma omp target
  {
    int p = 20;
    printf("Target Master Thread: Hello from %d\n", omp_get_thread_num());
    #pragma omp parallel for
    for (int i = 0; i < 2; i++) {
      int d = 0;
      if(verbose)
        printf("First Target Parallel: Hello from %d %p %p\n", omp_get_thread_num(), &d, &p);
      else
        printf("First Target Parallel: Hello.\n");
    }
    if(verbose)
      printf("End Parallel Master Thread: Hello from %d %p\n", omp_get_thread_num(), &p);
    else
      printf("End Parallel Master Thread: Hello from %d\n", omp_get_thread_num());
    #pragma omp parallel for
    for (int i = 0; i < 2; i++) {
      printf("Second Target Parallel: Hello.\n");
    }
  }
  }
  return 0;
}
