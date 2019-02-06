/* case from t-cricial */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#pragma omp declare target
const int bs = 1024;
const int nb = 128;
#pragma omp end declare target
const int X_VAL = 99;
const int Y_VAL = 11;

int main()
{
  int failures = 0;
  long cpuExec = 0;
#pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }

  // Checking team-level swap doesn't currently work on host.
  if (!cpuExec) {

    // Initialise.
    int *x = (int*)malloc(sizeof(int)*nb);
    int *y = (int*)malloc(sizeof(int)*nb);
    for(int ii = 0; ii < nb; ++ii) {
      x[ii] = X_VAL;
      y[ii] = Y_VAL;
    }
   int failures=0;
    /// Test team-level dependencies with increment
#pragma omp target map(tofrom: x[:nb], y[:nb])
#pragma omp teams num_teams(nb) thread_limit(bs)
#pragma omp distribute parallel for
    for(int ii = 0; ii < nb*bs; ++ii) {
#pragma omp critical
      {
        // Perform swap.
        const int temp = y[omp_get_team_num()];
        y[omp_get_team_num()] = x[omp_get_team_num()] + 1;
        x[omp_get_team_num()] = temp;
      }
    }

    // Validate.
    failures = 0;
    const int xcheck = X_VAL + (bs/2);
    const int ycheck = Y_VAL + (bs/2);
    for(int ii = 0; ii < nb; ++ii)
      failures += (x[ii] != xcheck || y[ii] != ycheck);
    if(failures)
      printf("failed %d times\n", failures);
    else
      printf("Succeeded\n");

     
  } else {// if !cpuExec
    printf("execution in CPU path\n");
  }

  return failures?1:0;
}

