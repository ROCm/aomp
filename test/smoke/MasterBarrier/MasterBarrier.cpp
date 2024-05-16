#include <stdio.h>
#include <omp.h>

#define N     100000
#define VALUE 123456

int main(int argc, char *argv[]) {

  int host[N] = {0x0};

  for(int dev = 0; dev < omp_get_num_devices(); ++dev)
  {
    #pragma omp target device(dev) map(from:host[0:N]) 
    {
      [[clang::loader_uninitialized]] static int A;
      #pragma omp allocate(A) allocator(omp_pteam_mem_alloc)
      A = VALUE;
      #pragma omp barrier

      #pragma omp parallel for num_threads(N)
      for(int i = 0 ; i<N ; i++) {
        host[i] = A;
      }
    }

    for (int i = 0; i < N;  ++i) {
      if (host[i] != VALUE) {
        printf("Failed on device %d host[%d]: %d (instead of %d)\n",dev,i,host[i],VALUE);
        return 1;
      }
    }
  }

  printf("OK\n");

  return 0;
}


