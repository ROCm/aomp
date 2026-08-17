#include <iostream>
#include <omp.h>

#define N 1000

template<typename T>
int compute_min_max() {
  T min_t = 1000;
  T max_t = 0;
  T *arr_t = new T[N];
  for (int i = 0; i < N; i++)
    arr_t[i] = i; // no overflow
#pragma omp target data map(to : arr_t[0 : N])
  {
#pragma omp target teams distribute parallel for reduction(min : min_t)
    for (int j = 0; j < N; j = j + 1)
      min_t = __builtin_fmin(min_t, arr_t[j]);

#pragma omp target teams distribute parallel for reduction(max : max_t)
    for (int j = 0; j < N; j = j + 1)
      max_t = __builtin_fmax(max_t, arr_t[j]);
  }
  delete[] arr_t;
  std::cout << "min_t = " << min_t << " max_t = " << max_t << "\n";
  int rc = 0;
  if (min_t != 0) {
    std::cout << "Failed min: expected 0\n";
    rc = 1;
  }
  T expected_max_t = 999;
  if (max_t != expected_max_t) {
    std::cout << "Failed max: expected " << expected_max_t << " \n";
    rc = 1;
  }
  return rc;
}

int main()
{
  int rc = 0;
  if (compute_min_max<short>())
    rc = 1;
  if (compute_min_max<ushort>())
    rc = 1;
  if (compute_min_max<int>())
    rc = 1;
  if (compute_min_max<unsigned int>())
    rc = 1;
  if (compute_min_max<long long>())
    rc = 1;
  if (compute_min_max<unsigned long long>())
    rc = 1;
  if (compute_min_max<float>())
    rc = 1;
  if (compute_min_max<double>())
    rc = 1;

  if (!rc)
    std::cout << "Success\n";
  else
    std::cout << "Failed\n";
  return rc;
}

// Cross-team reductions are emitted as plain SPMD kernels (SGN:2); the
// downstream Xteam reduction execution mode (SGN:8) has been removed.
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
