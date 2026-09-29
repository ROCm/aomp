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

#define N 1021

extern void hiptarg_test(int *a, int *c, int n);
extern void omptarg_test(int *b, int *c, int n);

void init_test(int *a, int *b, int n) {
  for(int i = 0; i < n; i++) {
    a[i] = i;
    b[i] = i+1;
  }
}

int check_test(int *a, int *b, int *c, int n) {
  int err = 0;
  for(int i = 0; i < n; i++)
    if(c[i] != a[i] + b[i]) {
      err++;
      printf("%d: c = %d (expected %d)\n", i, c[i], a[i]+b[i]);
      if (err > 10) return err;
  }   

  return err;
}

int main() {
  int *a = (int*) malloc(N*sizeof(int));
  int *b = (int*) malloc(N*sizeof(int));
  int *c = (int*) malloc(N*sizeof(int));

  init_test(a, b, N);
  hiptarg_test(a, c, N);
  omptarg_test(b, c, N);
  return check_test(a, b, c, N);
}
