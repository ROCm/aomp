#include <stdio.h>
#include <omp.h>
int main(void) {
   #pragma omp target
      printf("Hello3. Hello GPU world. Is this running on a CPU? %d\n", omp_is_initial_device());
}
