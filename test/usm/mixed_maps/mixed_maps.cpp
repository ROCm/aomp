#include <cstdlib>
#include <cstdio>

#include <omp.h>

#pragma omp requires unified_shared_memory

#define N 1024
#define K 123.456

void init(int n, double *a, double *b, double *c) {
  for(int i = 0; i < n; i++) {
    a[i] = 0;
    b[i] = i;
    c[i] = 2*i;
  }
}

int check(int n, double *a, double *b, double *c, const double k) {
  int err = 0;
  for(int i = 0; i < n; i++)
    if (a[i] != b[i]+c[i]+k) {
      err++;
      printf("Error at %d: a[%d] = %lf, expected %lf\n", i, i, a[i], b[i]+c[i]+k);
      if (err > 10) return err;
    }
  return err;
}

int main() {
  int n = N;
  int err = 0;
  double *a, *b, *c;

  {
    // all memory is mapped on target
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);

    #pragma omp target teams distribute parallel for map(from: a[:n]) map(to: b[:n], c[:n])
    for(int i = 0; i < n; i++)
      a[i] = b[i] + c[i];

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  {
    // all memory is mapped with enter data
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);
    #pragma omp target enter data map(alloc: a[:n]) map(to: b[:n], c[:n])

    #pragma omp target teams distribute parallel for
    for(int i = 0; i < n; i++)
      a[i] = b[i] + c[i];

    #pragma omp target exit data map(from: a[:n]), map(delete: b[:n], c[:n])

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  {
    // only first allocation is mapped
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);

    #pragma omp target teams distribute parallel for map(from: a[:n])
    for(int i = 0; i < n; i++)
      a[i] = b[i] + c[i];

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  {
    // no memory is mapped
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);

    #pragma omp target teams distribute parallel for
    for(int i = 0; i < n; i++)
      a[i] = b[i] + c[i];

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  {
    // only first allocation is mapped, and we are using omp_target_alloc additional memory
    a = new double[n];
    b = new double[n];
    c = new double[n];
    double *k = (double *) omp_target_alloc(n*sizeof(double), /*device=*/  0);

    init(n, a, b, c);
    for(int i = 0; i < n; i++) k[i] = K;

    #pragma omp target teams distribute parallel for map(from: a[:n])
    for(int i = 0; i < n; i++)
      a[i] = b[i] + c[i] + k[i];

    if ((err = check(n, a, b, c, K)))
      return err;

    delete a;
    delete b;
    delete c;
    omp_target_free(k, /*device=*/0);
  }

  {
    // map only half of two allocations
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);

    #pragma omp target teams distribute parallel for map(from: a[:n/2]) map(to: b[:n/2])
    for(int i = 0; i < n; i++)
      a[i] = b[i] + c[i];

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  {
    // map only one allocation and remap it in successive target region
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);

    #pragma omp target teams distribute parallel for map(from: a[:n])
    for(int i = 0; i < n; i++) {
      a[i] = b[i] + c[i] -1.0;
      b[i] = -1.0;
      c[i] = -2.0;
    }

    #pragma omp target teams distribute parallel for map(tofrom: a[:n], b[:n], c[:n])
    for(int i = 0; i < n; i++) {
      a[i] += 1.0;
      b[i] += i+1.0;
      c[i] += 2*i+2.0;
    }

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  {
    // partial map and from a struct
    struct PTRs {
      double *t_a, *t_b, *t_c;
    };
    a = new double[n];
    b = new double[n];
    c = new double[n];

    init(n, a, b, c);

    PTRs T;
    T.t_a = a;
    T.t_b = b;
    T.t_c = c;

    #pragma omp target teams distribute parallel for map(from: T.t_a[:n])
    for(int i = 0; i < n; i++) {
      T.t_a[i] = T.t_b[i] + T.t_c[i];
    }

    if ((err = check(n, a, b, c, 0.0)))
      return err;

    delete a;
    delete b;
    delete c;
  }

  return 0;
}
