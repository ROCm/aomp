#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <omp.h>

// start of saxpy header
void   saxpy(int, float, float *, float *);
void amdgcn_saxpy(int, float, float *, float *);
void nvptx_saxpy(int, float, float *, float *);

#pragma omp declare target
int return_code = 0 ;
#pragma omp end declare target

#pragma omp declare variant(nvptx_saxpy) \
  match(device = {arch(nvptx, nvptx64)}, implementation = {extension(match_any)})
#pragma omp declare variant( amdgcn_saxpy )  \
  match(device = {arch(amdgcn)}, implementation = {extension(match_any)})
void saxpy(int n, float s, float *x, float *y)    // base function
{
   printf("saxpy: Running on host . IsHost:%d\n", omp_is_initial_device());
   return_code=1;
   #pragma omp parallel for
   for(int i=0; i<n; i++) y[i] = s*x[i] + y[i];
}
void amdgcn_saxpy(int n, float s, float *x, float *y)    //function variant
{
   return_code=0;
   printf("amdgcn_saxpY: Running on amdgcn device. IsHost:%d\n", omp_is_initial_device());
   #pragma omp teams distribute parallel for
   for(int i=0; i<n; i++) { y[i] = s*x[i] + y[i]; }
}
void nvptx_saxpy(int n, float s, float *x, float *y)    //function variant
{
   return_code=0;
   printf("nvptx_saxpy: Running on nvptx device. IsHost:%d\n",omp_is_initial_device());
   #pragma omp teams distribute parallel for
   for(int i=0; i<n; i++) y[i] = s*x[i] + y[i];
}
//  end of saxpy header

#define N 128
#define THRESHOLD  127
int main() {
   static float x[N],y[N] __attribute__ ((aligned(64)));
   float s=2.0;

   for(int i=0; i<N; i++){ x[i]=i+1; y[i]=i+1; } // initialize
 
   printf("Calling saxpy with high device threshold\n");
   #pragma omp target if (N>(THRESHOLD*2))
   saxpy(N,s,x,y);

   printf("Calling saxpy with low device threshold\n");
   #pragma omp target if (N>THRESHOLD)
   saxpy(N,s,x,y); // will set rc to 0 if run on device

   printf("y[0],y[N-1]: %5.0f %5.0f\n",y[0],y[N-1]); //output: y...   5  5000

   return return_code;
}

