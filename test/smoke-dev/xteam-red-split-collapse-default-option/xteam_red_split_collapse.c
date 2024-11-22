#include <stdio.h>
#include <omp.h>

int N = 100;

void init(double *sum, double c[][N]) {
  *sum = 0;
  for (int i = 0; i < N; ++i) {
    for (int k = 0; k < N; ++k) {
      c[i][k] = 0;
    }
  }
}

int check(double sum, double c[][N]) {
  return sum != 495000 || c[10][20] != 4950;
}

int main()
{
  double a[N];
  double c[N][N];

  for (int i=0; i<N; i++)
    a[i]=i;

  double sum;
  double tmp;

  fprintf(stderr, "Starting test 1 on target\n");
  init(&sum, c);
#pragma omp target teams private(tmp) map(tofrom:sum) 
#pragma omp distribute parallel for reduction(+:sum) collapse(2) 
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 2 on target\n");
  init(&sum, c);
#pragma omp target teams map(tofrom:sum) 
#pragma omp distribute parallel for reduction(+:sum) collapse(2) private(tmp)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 3 on target\n");
  init(&sum, c);
#pragma omp target map(tofrom:sum) private(tmp)
#pragma omp teams 
#pragma omp distribute parallel for reduction(+:sum) collapse(2)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 4 on target\n");
  init(&sum, c);
#pragma omp target map(tofrom:sum) 
#pragma omp teams private(tmp)
#pragma omp distribute parallel for reduction(+:sum) collapse(2)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 5 on target\n");
  init(&sum, c);
#pragma omp target map(tofrom:sum) 
#pragma omp teams 
#pragma omp distribute parallel for reduction(+:sum) collapse(2) private(tmp)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 6 on target\n");
  init(&sum, c);
#pragma omp target map(tofrom:sum) private(tmp)
#pragma omp teams private(tmp)
#pragma omp distribute parallel for reduction(+:sum) collapse(2) private(tmp)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 7 on target\n");
  init(&sum, c);
#pragma omp target map(tofrom:sum)
#pragma omp teams distribute parallel for reduction(+:sum) collapse(2) private(tmp)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  fprintf(stderr, "Starting test 8 on target\n");
  init(&sum, c);
#pragma omp target teams distribute parallel for reduction(+:sum) collapse(2) private(tmp)
  for (int j = 0; j< N; j=j+1) {
    for (int i = 0; i < N; ++i) {
      sum += a[i];
      tmp = 0;
      for (int k = 0; k < N; ++k) {
	tmp += a[k];
      }
      c[i][j] = tmp;
    }
  }
  if (check(sum, c))
    return 1;

  printf("Success\n");
  return 0;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8


