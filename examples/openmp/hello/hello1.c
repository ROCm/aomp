#include <stdio.h>
#include <omp.h>
int main(void) {
   printf("Hello1. Hello CPU world. Is this running on a CPU? %d\n", omp_is_initial_device());
}
