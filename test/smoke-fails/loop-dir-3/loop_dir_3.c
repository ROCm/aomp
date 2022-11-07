#include <stdio.h>
#include <omp.h>

int main() {
  int i;
  int j;
  int sum[10][10] = {200};
  int exp_sum[10][10] = {200};
  int fail = 0;

  for(i=0; i<10; i++)
    for(j=0; j<10; j++)
      sum[i][j] = exp_sum[i][j] = i * j;

  #pragma omp target teams loop reduction(+:sum) collapse(2) \
      bind(parallel) order(concurrent) lastprivate(j) map(tofrom:sum)
  for(i=0; i<10; i++)
    for(j=0; j<10; j++)
      sum[i][j] -= i;

  #pragma omp target teams distribute parallel for reduction(+:exp_sum) \
      collapse(2) order(concurrent) map(tofrom:exp_sum)
  for(i=0; i<10; i++)
    for(j=0; j<10; j++)
      exp_sum[i][j] -= i;

  for(i=0; i<10; i++)
    for(j=0; j<10; j++) {
      if (sum[i][j] != exp_sum[i][j]) {
        printf("Wrong answer for sum[%d][%d]: %d\n", i, j, exp_sum[i][j]);
        fail++;
        break;
      }
    }

  if (!fail)
    printf("Success\n");

  return fail;
}
