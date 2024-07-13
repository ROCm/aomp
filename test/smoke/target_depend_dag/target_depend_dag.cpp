#include <stdio.h>
#include <omp.h>

#define N 12345

// create a DAG of nowait target tasks for two devices in a single thread; these can be then executed by  multiple threads

int main() {
  int n = N;
  int *a = new int[n];
  int *b = new int[n];
  int *c = new int[n];

  char t0_0, t0_1, t0_2, t0_3;
  char t1_0, t1_1;
  char t2_0, t2_1;

  int err = 0;
  #pragma omp parallel
  {
    #pragma omp masked
    {
      #pragma omp target enter data map(to: a[0:n]) nowait depend(out:t0_0) device(0)

      #pragma omp target parallel for nowait depend(in:t0_0) depend(out:t0_1) device(0)
      for(int i = 0; i < n; i++)
	a[i] = i;

      #pragma omp target exit data map(from:a[0:n]) nowait depend(in:t0_1) depend(out:t0_2) device(0)


      #pragma omp target enter data map(to:b[0:n]) nowait depend(out:t1_0) device(1)

      #pragma omp target parallel for nowait depend(in:t1_0) depend(out:t1_1) device(1)
      for(int i = 0; i < n; i++)
	b[i] = 2*i;

      #pragma omp target enter data map(to:c[:n]) nowait depend(out:t2_0) device(1)

      #pragma omp target enter data map(to:a[0:n]) nowait depend(in:t0_2) depend(out:t0_3) device(1)

      #pragma omp target parallel for nowait depend(in:t0_3,t1_1,t2_0) depend(out:t2_1) device(1)
      for(int i = 0; i < n; i++) {
	c[i] = a[i] + b[i];
      }

      #pragma omp target exit data map(from:c[:n]) nowait depend(in:t2_1) device(1)
    }

    // thread with an empty task queue (all except the one selected by masked) are able to execute target tasks
  } // task synch point: after this point, all pending tasks will be completed

  // check results
  for(int i = 0; i < n; i++) {
    if (c[i] != (double) 2*i+i) {
      err++;
      printf("%d: got %d, expected %d\n", i, c[i], 2*i+i);
      if (err > 10)
	break;
    }
  }

	delete[] a;
  delete[] b;
  delete[] c;

  if (!err) printf("Success\n");

  return err;
}
