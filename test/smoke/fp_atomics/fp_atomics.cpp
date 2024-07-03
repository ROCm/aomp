#include <cstdio>
#include <omp.h>

#define N 1024

#pragma omp declare target
template <typename T> __attribute__((used)) void fast(T *addr, T val) {
#pragma omp atomic hint(AMD_fast_fp_atomics)
  *addr += val;
}
#pragma omp end declare target

#pragma omp declare target
template <typename T> __attribute__((used)) void safe(T *addr, T val) {
#pragma omp atomic hint(AMD_safe_fp_atomics)
  *addr += val;
}
#pragma omp end declare target

template <typename T> void init(T *a, int n) {
#pragma omp parallel for
  for (int i = 0; i < n; i++)
    a[i] = (T)i;
}

template <typename T> int test_em() {
  int n = N;
  T expect = (T)(((T)n - 1) * (T)n) / 2.0;
  T *a = new T[n];
  T red = 0.0;

  init(a, n);

  // simplest fast case
  #pragma omp target teams distribute parallel for \
    map(to: a[:n]) map(tofrom: red)
  for (int i = 0; i < n; i++) {
    #pragma omp atomic hint(AMD_fast_fp_atomics)
    red += a[i];
  }

  if (red != expect) {
    printf("Error: simplest sp fast, got %f should be %f\n", red, expect);
    return 1;
  }

  red = 0.0;

  // global variable hidden in a function, fast
  #pragma omp target teams distribute parallel for \
  map(to: a[:n]) map(tofrom: red)
  for (int i = 0; i < n; i++)
    fast<T>(&red, a[i]);

  if (red != expect) {
    printf("Error: behind function sp fast, got %f should be %f\n", red,
           expect);
    return 1;
  }

  red = 0.0;

  // simplest safe case
  #pragma omp target teams distribute parallel for \
    map(to: a[:n]) map(tofrom: red)
  for (int i = 0; i < n; i++) {
    #pragma omp atomic hint(AMD_safe_fp_atomics)
    red += a[i];
  }

  if (red != expect) {
    printf("Error: simplest sp safe, got %f should be %f\n", red, expect);
    return 1;
  }

  red = 0.0;

  // global variable hidden in a function, safe
  #pragma omp target teams distribute parallel for \
    map(to: a[:n]) map(tofrom: red)
  for (int i = 0; i < n; i++)
    safe<T>(&red, a[i]);

  if (red != expect) {
    printf("Error: behind function sp safe, got %f should be %f\n", red,
           expect);
    return 1;
  }

  red = 0.0;

  // LDS, fast
  #pragma omp target teams num_teams(1) map(to : a[:n]) map(tofrom : red)
  { // no distribute construct, this works with a single team only
    T red_sh;
    #pragma allocate(red_sh) allocator(omp_pteam_mem_alloc)
    // static __attribute__((address_space(3))) T red_sh;
    red_sh = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      #pragma omp atomic hint(AMD_fast_fp_atomics)
      red_sh += a[i];
    }
    red = red_sh;
  }

  if (red != expect) {
    printf("Error: LDS version, got %f should be %f\n", red, expect);
    return 1;
  }

  red = 0.0;

  // LDS, safe
  #pragma omp target teams num_teams(1) map(to : a[:n]) map(tofrom : red)
  { // no distribute construct, this works with a single team only
    T red_sh;
    #pragma allocate(red_sh) allocator(omp_pteam_mem_alloc)
    red_sh = 0.0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
      #pragma omp atomic hint(AMD_safe_fp_atomics)
      red_sh += a[i];
    }
    red = red_sh;
  }

  if (red != expect) {
    printf("Error: LDS version, got %f should be %f\n", red, expect);
    return 1;
  }

  delete[] a;
  return 0;
}

int main() {
  int err = test_em<float>();
  if (err)
  return err;
  err = test_em<double>();
  return err;
}
