#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 10;
  int M = 10;

  double a[M*N];
  double a_ref[M*N];
  double b[M*N];

  for (int i=0; i<M*N; i++) {
    a[i] = 0;
    b[i] = i;
  }

  // HOST calculation of a_ref:
  int NN = 10;
  double A[NN];
  for(int i=0; i<M*N; i++) {
    double red_A = 0.0;
    double value = b[i];
    a_ref[i] = value;

    for (int k=0; k<NN; k++) {
      A[k] = k;
    }

    for(int k=0; k<NN; k++) {
      red_A += A[k];
    }

    a_ref[i] += red_A;
  }

  // DEVICE calculation of a:
  #pragma omp target teams distribute parallel for map(to:b[:M*N]) map(from:a[:M*N])
  for(int i=0; i<M*N; i++) {
    int NN = 10;
    double A[NN];
    double red_A = 0.0;
    double value = b[i];
    a[i] = value;

    for (int k=0; k<NN; k++) {
      A[k] = k;
    }

    for(int k=0; k<NN; k++) {
      red_A += A[k];
    }

    a[i] += red_A;
  }

  // Compare host and device results:
  for (int i=0; i<M*N; i++) {
    if (a_ref[i] != a[i] ) {
      printf("Wrong value: a[%d] = %f\n", i, a[i]);
      return 1;
    }
  }

  printf("Success\n");
  return 0;
}
