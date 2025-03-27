#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* Scan and min/max not yet supported */

#define N 2000000

int run_test() {
  int rc = 0;
  float *in = new float[N];
  float *out1 = new float[N]; // For inclusive scan
  float *out2 = new float[N]; // For exclusive scan

  for (int i = 0; i < N; i++) {
    in[i] = 10;
    out1[i] = 0;
  }

  float max1 = 0;
  float min1 = 3000000;

#pragma omp target teams distribute parallel for reduction(inscan, min : min1) map(tofrom: in[0:N], out1[0:N])
  for (int i = 0; i < N; i++) {
    min1 = fminf(min1, in[i]); // input phase
#pragma omp scan inclusive(min1)
    out1[i] = min1; // scan phase
  }

  float checksum = 3000000;
  for (int i = 0; i < N; i++) {
    checksum = fminf(checksum, in[i]);
    if (checksum != out1[i]) {
      printf("Inclusive Scan: Failure. Wrong Result at %d. Exiting...\n", i);
      rc = 1;
      break;
    }
  }
  if (!rc)
    printf("Inclusive Scan: Success!\n");
  if (min1 != 10) {
    printf("Min failed: found %f, expected 10\n", min1);
    rc = 1;
  }
  delete [] out1;

#pragma omp target teams distribute parallel for reduction(inscan, max: max1) map(tofrom: in[0:N], out2[0:N])
  for (int i = 0; i < N; i++) {
    out2[i] = max1; // scan phase
#pragma omp scan exclusive(max1)
    max1 = fmaxf(max1, in[i]); // input phase
  }

  checksum = 0;
  for (int i = 0; i < N; i++) {
    if (checksum != out2[i]) {
      printf("Exclusive Scan: Failure. Wrong Result at %d. Exiting...\n", i);
      rc = 1;
      break;
    }
    checksum = fmaxf(checksum, in[i]);
  }
  if (!rc)
    printf("Exclusive Scan: Success!\n");
  if (max1 != 10) {
    printf("Max failed: found %f, expected 10\n", max1);
    rc = 1;
  }
  delete [] in;
  delete [] out2;

  return rc;
}

int main() {
  int rc = run_test();
  return rc;
}
