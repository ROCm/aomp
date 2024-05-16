#include <stdio.h>

#define N 12893273

void init(float *v, int n) {
  for(int i = 0; i < n; i++)
    v[i] = i;
}

int output(float *v, int n) {
  int err = 0;
  for(int i = 0; i < n; i++) {
    if (v[i] != (float)(i+i)) {
      err++;
      printf("Error at %d, got %f, expected %f\n", i, v[i], (float)(i*i));
      if (err > 10) break;
    }
  }

  return err;
}

int vec_mult(int n)
{
  int i;
  float *p = new float[n];
  float *v1 = new float[n];
  float *v2 = new float[n];
  int err = 0;

  #pragma omp parallel num_threads(2)
  {
    #pragma omp single
    {
      #pragma omp task depend(out:v1)
      init(v1, n);

      #pragma omp task depend(out:v2)
      init(v2, n);

      #pragma omp target nowait depend(in:v1,v2) depend(out:p)	\
	map(to:v1[:n],v2[:n]) map( from: p[:n])
      {
	#pragma omp parallel for private(i)
	for (i=0; i<n; i++)
	  p[i] = v1[i] + v2[i];
      }
      #pragma omp task depend(in:p)
      err = output(p, n);
    }
  }

  return err;
}

int main() {
  int n = N;
  return vec_mult(n);
}
