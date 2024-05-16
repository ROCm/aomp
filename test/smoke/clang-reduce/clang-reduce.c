#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
  double x[8192];
  double y[8192];
  double cgdot;
  int i,N;
  N=8192;
  for (i=0; i<8192; i++) {
    x[i] = i * 0.5;
    y[i] = i * 0.5;
  }
  cgdot = 0.0;
#pragma omp target teams distribute parallel for reduction(+:cgdot)
  for (int i = 0; i < N; i++) {
    cgdot=cgdot+x[i]*y[i];
  }
  return 0;
}
