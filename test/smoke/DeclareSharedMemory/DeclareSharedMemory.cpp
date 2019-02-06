#include <stdio.h>
#include <omp.h>

#define N     10
#define VALUE 123456

int main(int argc, char *argv[]) {

  int host[N] = {0x0};

  for(int dev = 0; dev < omp_get_num_devices(); ++dev)
  {
    #pragma omp target device(dev) map(from:host[0:N])
    {
      static __attribute__((address_space(3))) int A;
      A = VALUE;

      #pragma omp parallel num_threads(N)
      {
        //printf("%d\n",A);
        int i = omp_get_thread_num();
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


