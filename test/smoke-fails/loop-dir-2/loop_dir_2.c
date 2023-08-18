#include <stdio.h>
#include <omp.h>

int main() {
  int k;
  int sum = 0;
  int fail = 0;

  // If a loop construct is not nested inside another OpenMP construct then
  // the bind clause must be present. The following 'loop' directive is
  // equivalent to a 'for' (a worksharing-loop).
  #pragma omp loop reduction(+:sum) bind(parallel)
  for(k=0; k<10; k++) {
    sum += k;
  }
  if (sum != 45) {
    fail++;
    printf("Wrong result for 'loop': sum is %d\n", sum);
  }

  sum = 0;
  #pragma omp parallel loop reduction(+:sum)
  for(k=0; k<10; k++) {
    sum += k;
  }
  if (sum != 45) {
    fail++;
    printf("Wrong result for 'parallel loop': sum is %d\n", sum);
  }

  sum = 0;
  #pragma omp target teams loop reduction(+:sum)
  for(k=0; k<10; k++) {
    sum += k;
  }
  if (sum != 45) {
    fail++;
    printf("Wrong result for 'target teams loop': sum is %d\n", sum);
  }

  sum = 0;
  #pragma omp target parallel loop reduction(+:sum)
  for(k=0; k<10; k++) {
    sum += k;
  }
  if (sum != 45) {
    fail++;
    printf("Wrong result for 'target parallel loop': sum is %d\n", sum);
  }

  if (!fail)
    printf("Success\n");

  return fail;
}
