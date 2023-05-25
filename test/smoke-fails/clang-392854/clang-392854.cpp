#include <omp.h>
#ifndef _OPENMP
#define _OPENMP
//#error "_OPENMP NOT defined"
#endif

#include <iostream>

int main(void) {
  int num_threads;
#pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  std::cout << "OpenMP has created us " << num_threads << " threads." << std::endl;
#ifndef _OPENMP
  std::cout << "...but _OPENMP is NOT defined." << std::endl;
#else
  std::cout << "...and _OPENMP is defined." << std::endl;
#endif
  return 0;
}
