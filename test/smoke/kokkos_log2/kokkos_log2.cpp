#include <cmath>
#include <algorithm>
#include <omp.h>
#include <stdio.h>

int main(int argc, char **argv)
{
  static_assert(std::is_same<decltype(std::log2((int)0)), double>::value, "");
  printf("Pass\n");
  return 0;
}
