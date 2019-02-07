#include <stdlib.h>
#include <stdio.h>

/// Two methods store in different compilation units, results in a compile-time
//  error. Duplicate .omp_offloading.entry
void test_comp_unit_1(const int niters, double* a);
void test_comp_unit_2(const int niters, double* a);

int main()
{
  const int niters = 10;
  double* a = (double*)malloc(sizeof(double)*niters);

  #pragma omp target data map(from:a[:niters])
  {
    test_comp_unit_1(niters, a);
    test_comp_unit_2(niters, a);
  }

  double res = 0.0;
  for(int ii = 0; ii < niters; ++ii)
  {
    res += a[ii];
  }
  
  printf("--> %s <--\n",(res < 90.001 && res > 89.999) ? "success" : "error");
  return 0;
}

/// Presumably creates an .omp_offloading.entry
void test_comp_unit_1(const int niters, double* a)
{
  #pragma omp target 
    for(int ii = 0; ii < niters; ++ii)
      a[ii] = (double)ii;
}