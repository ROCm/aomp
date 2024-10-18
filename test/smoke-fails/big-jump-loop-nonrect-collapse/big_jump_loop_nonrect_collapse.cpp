#include <cstdio>
#include <cstring>

const int N = 256;

int tri1 (double *arr) {
  int errors = 0;

#pragma omp target teams distribute parallel for collapse(2) map(tofrom: arr[0:N*N*2])
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < j; i++) {
      arr[j * N * 2 + i]++;
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N * 2; i++) {
      if (i >= N)
        errors += arr[j * N * 2 + i] != 0;
      else if (i < j)
        errors += arr[j * N * 2 + i] != 1;
      else
        errors += arr[j * N * 2 + i] != 0;
    }
  }
  return errors;
}

int tri2 (double *arr) {
  int errors = 0;

#pragma omp target teams distribute parallel for collapse(2) map(tofrom: arr[0:N*N*2])
  for (int j = 0; j < N; j++) {
    for (int i = j; i < N; i++) {
      arr[j * N * 2 + i]++;
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N * 2; i++) {
      if (i >= N)
        errors += arr[j * N * 2 + i] != 0;
      else if (i >= j)
        errors += arr[j * N * 2 + i] != 1;
      else
        errors += arr[j * N * 2 + i] != 0;
    }
  }
  return errors;
}

int tri3 (double *arr) {
  int errors = 0;

#pragma omp target teams distribute parallel for collapse(2) map(tofrom: arr[0:N*N*2])
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < j; i++) {
      arr[i * N + j]++;
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N * 2; i++) {
      if (i >= N)
        errors += arr[i * N + j] != 0;
      else if (i < j)
        errors += arr[i * N + j] != 1;
      else
        errors += arr[i * N + j] != 0;
    }
  }
  return errors;
}

int tri4 (double *arr) {
  int errors = 0;

#pragma omp target teams distribute parallel for collapse(2) map(tofrom: arr[0:N*N*2])
  for (int j = 0; j < N; j++) {
    for (int i = j; i < N; i++) {
      arr[i * N + j]++;
    }
  }

  for (int j = 0; j < N; j++) {
    for (int i = 0; i < N * 2; i++) {
      if (i >= N)
        errors += arr[i * N + j] != 0;
      else if (i >= j)
        errors += arr[i * N + j] != 1;
      else
        errors += arr[i * N + j] != 0;
    }
  }
  return errors;
}

int main() {
  double *arr = new double[N*N*2];
  int errors = 0;

  memset (arr, 0, sizeof (double) * (N * N * 2));

  errors += tri1(arr);

  memset (arr, 0, sizeof (double) * (N * N * 2));

  errors += tri2(arr);

  memset (arr, 0, sizeof (double) * (N * N * 2));

  errors += tri3(arr);

  memset (arr, 0, sizeof (double) * (N * N * 2));

  errors += tri4(arr);

  delete[] arr;

  if (errors)
    fprintf (stderr, "FAILED: errors=%d\n", errors);
  else
    fprintf (stderr, "SUCCESS\n");

  return errors != 0;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:5
/// CHECK: SUCCESS
