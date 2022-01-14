#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#ifndef DSIZE
const uint64_t M = 1000000000;  // table size = 20 million
#else
const uint64_t M = DSIZE;
#endif


int main(int argc, char **argv) {
  struct timespec t0,t1,t2,t3,t4,t5;
  float *W = new float[M];
  double m;

  fprintf(stderr, "Starting omp_band DIZE=%lu\n",M);
  clock_gettime(CLOCK_REALTIME, &t1);
  #pragma omp target data map(tofrom: W[:M])
  {
   clock_gettime(CLOCK_REALTIME, &t2);
   m = (t2.tv_sec - t1.tv_sec) + (t2.tv_nsec - t1.tv_nsec)/1e9;
   fprintf(stdout, "Time %f for copy host to device\n", m);
   fprintf(stderr, "%f GBytes/sec\n",  M*4/m/(1000*1000*1000));
   #pragma omp target teams
     #pragma omp distribute parallel for
     for (uint64_t sample=0; sample <100; sample++) {
     } // end of 2nd kernel
   clock_gettime(CLOCK_REALTIME, &t3);
  } // End of data map region
  clock_gettime(CLOCK_REALTIME, &t4);
  m = (t4.tv_sec - t3.tv_sec) + (t4.tv_nsec - t3.tv_nsec)/1e9;
  for (int i=0;i<M;i++)
    W[i]+=2;
  fprintf(stdout, "Time %f for copy device to host\n", m);
  fprintf(stderr, "%f GBytes/sec\n",  M*4/m/(1000*1000*1000));
  // Check results here...
  delete[] W;
  fprintf(stdout, "\nPassed\n");
  return 0;
}
