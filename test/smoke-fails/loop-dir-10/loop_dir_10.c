#include <stdio.h>
#include <omp.h>

int N = 10000;

void foo(int i) {}

int main()
{
  int fail = 0;
  int a[N];
  int ea[N];
  int b[N];

  int i;
  for (i=0; i<N; i++) {
    b[i]=i;
    a[i]=0;
    ea[i]=0;
  }

  // Expected results.
  #pragma omp target teams distribute parallel for
  for (i=0; i < N; i++) {
    for (int j=0; j < N; j++)
      ea[i] = b[j] * N + j;
  }

  // Usual case, emit as 'target teams distribute parallel for'
  #pragma omp target teams loop
  for (i=0; i < N; i++) {
    for (int j=0; j < N; j++)
      a[i] = b[j] * N + j;
  }

  for (i=0; i<N; i++)
    if (a[i] != ea[i] ) {
      fail++;
      printf("'target teams loop' as 'distribute parallel for'\n");
      printf("  a[%d]=%d, expected ea[%d]=%d\n", i, a[i], i, ea[i]);
      break;
    }

  for (i=0; i<N; i++) {
    a[i]=0;
  }

  // Emit 'target teams distribute' due to 'loop bind(parallel)'
  #pragma omp target teams loop
  for (i=0; i < N; i++) {
    #pragma omp loop bind(parallel)
    for (int j=0; j < N; j++)
      a[i] = b[j] * N + j;
  }

  for (i=0; i<N; i++)
    if (a[i] != ea[i] ) {
      fail++;
      printf("'target teams loop' as 'distribute': loop bind(parallel)\n");
      printf("  a[%d]=%d, expected ea[%d]=%d\n", i, a[i], i, ea[i]);
      break;
    }

  for (i=0; i<N; i++) {
    a[i]=0;
  }

  // Emit 'target teams distribute' due to function call.
  #pragma omp target teams loop
  for (i=0; i < N; i++) {
    for (int j=0; j < N; j++) {
      foo(i);
      a[i] = b[j] * N + j;
    }
  }

  for (i=0; i<N; i++)
    if (a[i] != ea[i] ) {
      fail++;
      printf("'target teams loop' as 'distribute': function call\n");
      printf("  a[%d]=%d, expected ea[%d]=%d\n", i, a[i], i, ea[i]);
      break;
    }

  for (i=0; i<N; i++) {
    a[i]=0;
  }

  int nt = 0;
  // Compute expected results with equivalent combined directive.
  #pragma omp target teams distribute num_teams(32)
  for (i=0; i < N; i++) {
    if (!nt) nt = omp_get_num_teams();
    for (int j=0; j < N; j++)
      ea[j] = b[j] * N + nt;
  }

  nt = 0;
  // Emit 'target teams distribute' due to OpenMP API call.
  #pragma omp target teams loop num_teams(32)
  for (i=0; i < N; i++) {
    if (!nt) nt = omp_get_num_teams();
    for (int j=0; j < N; j++)
      a[j] = b[j] * N + nt;
  }

  for (i=0; i<N; i++)
    if (a[i] != ea[i] ) {
      fail++;
      printf("'target teams loop' as 'distribute': OpenMP API function call\n");
      printf("  a[%d]=%d, expected ea[%d]=%d\n", i, a[i], i, ea[i]);
      break;
    }

  if (!fail)
    printf("Success\n");
  return fail;
}
/// CHECK: DEVID:  0 SGN:5 ConstWGSize:256  args: 5 teamsXthrds:(  40X 256)
/// CHECK: DEVID:  0 SGN:5 ConstWGSize:256  args: 5 teamsXthrds:(  40X 256)
/// CHECK: DEVID:  0 SGN:3 ConstWGSize:257  args: 5 teamsXthrds:( 624X 256)
/// CHUCK: DEVID:  0 SGN:3 ConstWGSize:257  args: 5 teamsXthrds:( 624X 256)
/// CHECK: DEVID:  0 SGN:3 ConstWGSize:257  args: 6 teamsXthrds:(  32X 256)
/// CHECK: DEVID:  0 SGN:3 ConstWGSize:257  args: 6 teamsXthrds:(  32X 256)
/// CHECK: Success
