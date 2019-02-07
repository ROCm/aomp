// RUN: %libomptarget-compile-run-and-check-aarch64-unknown-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-powerpc64le-ibm-linux-gnu
// RUN: %libomptarget-compile-run-and-check-x86_64-pc-linux-gnu

#define IN_PARALLEL 0

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define N 1000

/*
 * Test if it is possible to:
 * 1. target enter data to depend 'in' and 'out'
 * 2. target exit data to depend 'in' and 'out'
 * 3. Mix target-based tasks with host tasks.
 */
int main(){
  int errors = 0;
  bool isHost = true;
  double sum = 0.0;
  double* h_array = (double *) malloc(N * sizeof(double));
  double* in_1 = (double *) malloc(N * sizeof(double));
  double* in_2 = (double *) malloc(N * sizeof(double));

#if IN_PARALLEL
#pragma omp parallel 
  {
    #pragma omp master
    {
#endif
    // host task
    #pragma omp task depend(out: in_1) shared(in_1)
    {
      for (int i = 0; i < N; ++i) {
        in_1[i] = 1;
      }
    }
    
    // host task
    #pragma omp task depend(out: in_2) shared(in_2)
    {
      for (int i = 0; i < N; ++i) {
        in_2[i] = 2;
      }
    }
    
    // target enter data
    #pragma omp target enter data nowait map(alloc: h_array[0:N]) map(to: in_1[0:N]) map(to: in_2[0:N]) depend(out: h_array) depend(in: in_1) depend(in: in_2) 
    
    // target task to compute on the device
    #pragma omp target nowait map(tofrom: isHost) depend(inout: h_array)
    {
      isHost = omp_is_initial_device();
      for (int i = 0; i < N; ++i) {
        h_array[i] = in_1[i]*in_2[i];
      }
    }
    
    // target exit data
    #pragma omp target exit data nowait map(from: h_array[0:N]) depend(inout: h_array) 
    
    // host task
    #pragma omp task depend(in: h_array) shared(sum, h_array)
    {
      // checking results
      for (int i = 0; i < N; ++i) {
        sum += h_array[i];
      }
    }
#if IN_PARALLEL
    } // master
  } // parallel
#else
  #pragma omp taskwait
#endif

  errors = 2.0*N != sum;

  if (!errors)
    printf("Test passed\n");
  else
    printf("Test failed on %s: sum = %g\n", (isHost ? "host" : "device"), sum);

  return errors;
}

