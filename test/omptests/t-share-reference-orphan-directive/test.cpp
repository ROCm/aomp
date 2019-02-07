#include <omp.h>
#include <stdio.h>

#pragma omp declare target
void kernel(int &a) {

  printf(" -> a = %d\n", a);

#pragma omp parallel num_threads(4)
  printf(" --> a = %d\n", a);
}
#pragma omp end declare target

int main(int argc, char *argv[]) {

  int b = 12345;

#pragma omp target map(to : b)
  {
    int &a = b;
    printf("a = %d\n", a);
    printf("b = %d\n", b);
    kernel(a);
  }

  return 0;
}
