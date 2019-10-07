#include <cmath>
#include <omp.h>
using namespace std;

int main(int argc, char **argv)
{
  #pragma omp target
  {
    float intpart, res;
    #pragma omp allocate(intpart) allocator(omp_thread_mem_alloc)
    res = modff(1.1f, &intpart);
  }

  #pragma omp target
  {
    double intpart, res;
    #pragma omp allocate(intpart) allocator(omp_thread_mem_alloc)
    res = modf(1.1, &intpart);
  }
  return 0;
}
