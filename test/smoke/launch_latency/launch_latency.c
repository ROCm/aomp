/*
for C in {0..8}; do /home/amd/rocm/aomp/bin/clang  -O2  -target x86_64-pc-linux-gnu -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906  -DCOPY=$C launch_latency.c -o launch_latency  && ./launch_latency ; done
*/

#include <stdio.h>
#include <time.h>
#include <omp.h>

#define MAX_TEAMS 2048

#define TRIALS (1000)

int n =1024;

#ifndef COPY
#error "Expected COPY macro"
#endif
#define USE(X) asm volatile("// nothing":: "r"(&d##X) :"memory")

int main(void) {

  // cost in kernel time per object
  // 512k costs 0.42
  // 256k 0.30
  // 128k 0.23
  // 64k costs about 0.18
  // 32k costs about 0.17
  // 16k costs 0.16ms
  // 8k costs 0.15ms
  // 8byte costs 0.15ms
  struct payload_t
  {
    double x[1024*64];
  };
  struct payload_t d0,d1,d2,d3,d4,d5,d6,d7;
  struct timespec t0,t1,t2;

  fprintf(stderr,"map tofrom %u objects of size %zu\n", COPY, sizeof(d0));
  int fail = 0;
  int a = -1;
  //
  clock_gettime(CLOCK_REALTIME, &t0);
  #pragma omp target
  { //nothing
  }
  clock_gettime(CLOCK_REALTIME, &t1);
  double m = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec)/1e9;
  fprintf(stderr, "1st kernel Time %12.8fms\n", 1000*m);
  int j = 8;
    clock_gettime(CLOCK_REALTIME, &t1);
    for (int t = 0 ; t < TRIALS ; t++) {
#pragma omp target  map(tofrom: d0,d1,d2,d3,d4,d5,d6,d7)
      for (int k =0; k < n; k++)
      {
        #if 0 < COPY
        USE(0);
        #endif
        #if 1 < COPY
        USE(1);
        #endif
        #if 2 < COPY
        USE(2);
        #endif
        #if 3 < COPY
        USE(3);
        #endif
        #if 4 < COPY
        USE(4);
        #endif
        #if 5 < COPY
        USE(5);
        #endif
        #if 6 < COPY
        USE(6);
        #endif
        #if 7 < COPY
        USE(7);
        #endif
        // nothing
      }
    }
    clock_gettime(CLOCK_REALTIME, &t2);
    double t = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)/1e9;
    fprintf(stderr, "avg kernel Time %12.8fms TEAMS=%d\n", 1000*t/TRIALS, j);

    fprintf(stderr, "Est cost per object %12.8fms\n", 1000*t/TRIALS/(COPY));
    printf("Succeeded\n");

  return fail;
}

