// Test reduction on C++ reference.
#include <iostream>
#include <omp.h>

void compute_reduced_sum(int n, int *x) {
#pragma omp target teams distribute parallel for reduction(+ : x[0:1])
  for (int i = 0; i < n; ++i)
    *x += i;
}

int main()
{
  int n = 1000;
  int sum = 0;
  compute_reduced_sum(n, &sum);

  std::cout << "sum = " << sum << "\n";
  int rc = (sum != 499500);
  if (!rc)
    std::cout << "Success\n";
  else
    std::cout << "Failed\n";
  return rc;
}
