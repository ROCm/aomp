#include <stdio.h>
#include <omp.h>

int main() {
  constexpr int N = 10;
  float MTX[N][N];
  float EMTX[N][N];
  int fail = 0;

  for (auto i = 0; i < N; ++i)
    for (auto j = 0; j < N; ++j) {
      MTX[i][j] = i + j;
      EMTX[i][j] = MTX[i][j];
    }

  // Check use of 'loop' directive nested in a combined 'teams loop'
  // directive.
  #pragma omp teams loop reduction(*:MTX)
  for (auto i = 0; i < N; ++i) {
    #pragma omp loop
    for (auto j = 0; j < N; ++j)
      MTX[i][j] *= N;
  }

  #pragma omp teams distribute parallel for reduction(*:EMTX)
  for (auto i = 0; i < N; ++i) {
    for (auto j = 0; j < N; ++j)
      EMTX[i][j] *= N;
  }

  for (auto i = 0; i < N; ++i)
    for (auto j = 0; j < N; ++j) {
      if (MTX[i][j] != EMTX[i][j]) {
        printf("Wrong answer for MTX[%d][%d]: %f ", i, j, MTX[i][j]);
        printf("expected %f\n", EMTX[i][j]);
        ++fail;
        break;
      }
    }

  if (!fail)
    printf("Success\n");

  return fail;
}
