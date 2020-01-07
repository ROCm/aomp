// MIT License
//
// Copyright (c) 2017 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy,
// modify, merge, publish, distribute, sublicense, and/or sell copies
// of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <omp.h>
#include <stdio.h>
#include "matsup.h"

#define N 3
#define M 3
#define P 3
#define Z 10


void targetMatrixMul(int *pA, int *pB, int *pC, int ARows,
                          int ACols, int BCols) {
   int i,j,k;
#pragma omp target data map (to: pA[0:ARows*ACols],pB[0:ACols*BCols]) map (tofrom: pC[0:ARows*BCols])
#pragma omp target
#pragma omp teams distribute parallel for private(i,j,k)
//#pragma omp teams distribute parallel for collapse(2) private(i,j,k)
   for(i=0;i<ARows;i++) {
      for(j=0;j<BCols;j++) {
         for(k=0;k<ACols;k++) {
             pC[i*BCols +j] += pA[i * ACols + k] * pB[k * BCols + j];
         }
      }
   }
}

int omptarg_test() {
  int N_errors = 0;
  int N_target_errors = 0;
  int hostSrcMatA[N * M];
  int hostSrcMatB[M * P];
  int hostDstMat[N * P];

  //  Use openmp target region for matrix multtiply
  //  FIXME:  Put this in a loop like the above but use target data region to only 
  //          send the input matricies one time but check each destination. 
  //          However, do not wrap target regions in openmp tasks because we do not 
  //          know how threadsafe is the libomptarget. 
  clearMatrix(hostDstMat, N, P);
  printf("\n\nCalling targetMatrixMul \n");
  targetMatrixMul(hostSrcMatA, hostSrcMatB, hostDstMat, N, M, P);
  N_target_errors = matrixMul_check(hostSrcMatA, hostSrcMatB, hostDstMat, N, M, P);
  if (N_target_errors != 0) {
     printf("targetMatrixMul \t\t FAILED: %d Errors >>>\n", N_target_errors);
  } else printf("targetMatrixMul \t\t SUCCESSFUL >>>\n");

     printf("\nMultarget: ");
     printMatrix(hostDstMat, N, P);
     printf("\n");

  if(!(N_errors+N_target_errors))
    printf("\n%s\n" , "Success");
  else
    printf("\n%s\n", "Failed\n");

  return N_errors+N_target_errors;
}
