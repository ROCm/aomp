#include <stdio.h>
#include <omp.h>

#define N     1024
#define VALUE 123456

int main(int argc, char *argv[]) {

  int host[N] = {0x0};

  for(int dev = 0; dev < omp_get_num_devices(); ++dev)
  {
    int num_threads = -1;
    #pragma omp target device(dev) map(from:host[0:N]) map(from:num_threads)
    {
      static __attribute__((address_space(3))) int A;
      A = VALUE;

      #pragma omp parallel num_threads(N)
      {
	if (omp_get_thread_num() == 0)
	  num_threads = omp_get_num_threads();
        //printf("%d\n",A);
        int i = omp_get_thread_num();
        host[i] = A;
      }
    }

    // the runtime is free to choose the best number of threads <= num_threads value, related ICV, etc.
    for (int i = 0; i < num_threads;  ++i) {
      if (host[i] != VALUE) {
        printf("Failed on device %d host[%d]: %d (instead of %d)\n",dev,i,host[i],VALUE);
        return 1;
      }
    }
  }

  printf("OK\n");

  return 0;
}
