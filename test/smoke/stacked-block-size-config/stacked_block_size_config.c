#include <stdio.h>
#include <stdlib.h>

#define SMALL 1000
#define LARGE 1000000

int main() {
  double *x = (double *)malloc(sizeof(double)*SMALL);
  double *y = (double *)malloc(sizeof(double)*LARGE);

  for(int i = 0; i < SMALL; i++) {
    x[i] = 1.0;
  }

  for(int i = 0; i < LARGE; i++) {
    y[i] = 5.0;
  }

  #pragma omp target teams distribute parallel for map(tofrom: x[:SMALL])
  for(int i = 0; i<SMALL; i++) {
    x[i] += 3.0;
  }

  #pragma omp target teams distribute parallel for map(tofrom: y[:LARGE])
  for(int i = 0; i<LARGE; i++) {
    y[i] += 3.0;
  }

  printf("x[3] = %f\n", x[3]);
  printf("y[500000] = %f\n", y[500000]);

  free(x);
  free(y);

  /// The block size from the command line is not a user override: the low
  /// trip count rule still lowers the threads to the minimum.
  /// CHECK: DEVID:{{.*}}SGN:2 ConstWGSize:128  args: 2 teamsXthrds:({{ *}}63X  16) {{.*}}tripcount:1000 rpc:0
  /// CHECK: DEVID:{{.*}}SGN:2 ConstWGSize:128  args: 2 teamsXthrds:({{.*}}X 128) {{.*}}tripcount:1000000 rpc:0

  /// CHECK: x[3] = 4
  /// CHECK: y[500000] = 8

  return 0;
}

