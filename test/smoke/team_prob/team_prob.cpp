#include <stdio.h>
#include <omp.h>

/// REQUIRES: amdgpu-registered-target

// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple x86_64-unknown-unknown -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple amdgcn-amd-amdhsa -fopenmp-targets=amdgcn-amd-amdhsa -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s

#ifndef HEADER
#define HEADER

#define min(x, y) (((x) < (y)) ? (x) : (y))

float dotprod(float B[], float C[], int N, int block_size,
int num_teams, int block_threads) {
  float sum = 0.0;
  int i, i0;
  #pragma omp target map(to: B[0:N], C[0:N]) map(tofrom: sum)
  #pragma omp teams num_teams(num_teams) thread_limit(block_threads) reduction(+:sum)
  #pragma omp distribute
  for (i0=0; i0<N; i0 += block_size)
    #pragma omp parallel for reduction(+:sum)
    for (i=i0; i< min(i0+block_size,N); i++)
      sum += B[i] * C[i];
  return sum;
}

#endif
int main(void) {

  int fail = 0;

  return fail;
}

