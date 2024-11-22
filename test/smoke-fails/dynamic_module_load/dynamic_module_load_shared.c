#include <stdio.h>

int foo() {
#pragma omp target
  ;

  return 0;
}
