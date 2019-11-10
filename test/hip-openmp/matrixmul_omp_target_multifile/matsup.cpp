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


int matrixMul_check(int *matrixA, int *matrixB, int *matrixC, int ARows,
                     int ACols, int BCols) {
  int n_errors = 0;
  for (int j = 0; j < BCols; ++j) {
    for (int i = 0; i < ARows; ++i) {
      int value = 0;
      for (int k = 0; k < ACols; ++k) {
        value += matrixA[i * ACols + k] * matrixB[k * BCols + j];
      }
      if(matrixC[i * BCols + j] != value) {
        ++n_errors;
        printf("\tError: Matrices miscompare devC[%d,%d]-hostC: %d\n", i, j,
               matrixC[i * BCols + j] - value);
      }
    }
  }
  return n_errors;
}

void printMatrix(int *matrix, int Rows, int Cols) {
  for (int i = 0; i < Rows; ++i) {
    printf("\n[");
    bool first = true;
    for (int j = 0; j < Cols; ++j) {
      if (first) {
        printf("%d", matrix[i * Cols + j]);
        first = false;
      } else {
        printf(", %d", matrix[i * Cols + j]);
      }
    }
    printf("]");
  }
}

void randomizeMatrix(int *matrix, int Rows, int Cols) {
  for (int i = 0; i < Rows * Cols; ++i)
    matrix[i] = rand() % 10;
}

void clearMatrix(int *matrix, int Rows, int Cols) {
  for (int i = 0; i < Rows * Cols; ++i)
    matrix[i] = 0;
}

