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

#include <cstdio>
#include <cstdlib>
#include <omp.h>

#define N 100

void Print(int arr[], int n) {
  for (int i = 0; i < n; i++) {
    printf("\n%d", arr[i]);
  }
}

int main(int argc, char *argv[]) {

  int A[N], B[N], C[N];

  for (int i = 0; i < N; i++) {
    A[i] = 2 * (i + 1);
    B[i] = 3 * (i + 1);
  }

#pragma omp target map(to : A [0:N], B [0:N]) map(from : C [0:N])
  {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
      C[i + 10] = A[i] + B[i];
    }
  }

  Print(C, N);

  printf("\n");
  return 0;
}
