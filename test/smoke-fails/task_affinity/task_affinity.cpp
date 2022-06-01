#include<cstdio>

double * alloc_init_B(double *A, int N) {
  double *B = new double[N];
  for(int i = 0; i < N; i++)
    B[i] = A[i];
  return B;
}

void compute_on_B(double *B, int N) {
  for(int i = 0; i < N; i++)
    B[i] += 1.0; 
}

int check_B(double *B, int N) {
  int err = 0;
  for(int i = 0; i < N; i++)
    if (B[i] != (double)i+1.0) {
      err++;
      printf("Error at %d: got %lf, expected %lf\n", i, B[i], (double)i+1.0);
      if (err > 10) return err;
    }
  return err;
}

int main() {
  int N = 1024;
  double *A = new double[N];
  double *B;

  for(int i = 0; i < N; i++)
    A[i] = i;

  #pragma omp task depend(out:B) shared(B) affinity(A[0:N])
  {
    B = alloc_init_B(A,N);
  }
  #pragma omp task depend( in:B) shared(B) affinity(A[0:N])
  {
    compute_on_B(B,N);
  }

  #pragma opm taskwait

  return check_B(B, N);
}
