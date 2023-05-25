#include "hip/hip_runtime.h"
#include <stdio.h>
#include <vector>
#include <chrono>

__global__ void EmptyKernel() { }
 
int main() {
 
  fprintf(stderr, "Starting program\n");

  std::vector<size_t> testSize{
        1024
       };
 
   auto start = std::chrono::steady_clock::now();
for (int istep = 0; istep < testSize.size(); istep++){
  size_t n = testSize.at(istep);
  size_t blocksize = 1024;
  size_t gridsize = (size_t)ceil((double)n/blocksize);
 
  const int N = 4000000;
 
  float time, cumulative_time = 0.f;
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
 
  for (int i=0; i<N; i++) {
    hipLaunchKernelGGL((EmptyKernel), dim3(gridsize), dim3(blocksize), 0, 0 );
  }
}
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  fprintf(stderr,"Kernel time:  %3.5f secs \n", elapsed_seconds.count());
return 0;
}
