#include <stdio.h>
#include <omp.h>

#define N 10
#define TN 100
#define GPU_REPS 1000000000
#define TARGET_REPS 100000

void target_fn(int *a)
{
#pragma omp target map(a[0:TN-1]) 
    {
      for (int j = 0; j < TN; j++) a[j]*=a[j];
    }
}

int target_reps()
{
  int a[TN];

  omp_set_default_device(0);
  for (int i=0; i<TN; i++) a[i]=1;

  printf("target ...\n");
  for(unsigned long i=0; i < TARGET_REPS; i++) {  
    target_fn(a);
  }

  return a[0];
}


int gpu_reps()
{
  int a[N];

  for (int i=0; i<N; i++) a[i] = 1;

  printf("gpu ...\n");
#pragma omp target 
  {
   for(unsigned long i=0; i < GPU_REPS; i++) {  
    for (int j = 0; j < N; j++) a[j]*=a[j];
   }
  }

  return a[0];
}

int main()
{
  int target = target_reps();
  int gpu = gpu_reps();

  printf("target + gpu = %d\n", target + gpu);

  return 0;
}
