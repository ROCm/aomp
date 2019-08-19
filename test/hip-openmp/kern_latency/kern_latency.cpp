#include "hip/hip_runtime.h"
#include <stdio.h>
#include <vector>
 
__global__ void EmptyKernel() { }
 
int main() {
 
  fprintf(stderr, "Starting program\n");

  std::vector<size_t> testSize{
        1024
       ,1024*1024
       ,1024*1024*2
       ,1024*1024*4
       ,1024*1024*8
       ,1024*1024*16
       ,1024*1024*32
       ,1024*1024*64
       };
 
for (int istep = 0; istep < testSize.size(); istep++){
  size_t n = testSize.at(istep);
  size_t blocksize = 1024;
  size_t gridsize = (size_t)ceil((double)n/blocksize);
 
  // was: const int N = 100000;
  const int N = 20000;
 
 
  float time, cumulative_time = 0.f;
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
 
  for (int i=0; i<N; i++) {
    hipEventRecord(start, 0);
    hipLaunchKernelGGL((EmptyKernel), dim3(gridsize), dim3(blocksize), 0, 0 );
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&time, start, stop);
    cumulative_time = cumulative_time + time;
  }
  fprintf(stderr,"Kernel launch overhead time for n = %ld :  %3.5f ms \n", n, cumulative_time / N);

}
return 0;
}
