#include <stdio.h>
#define N 1024

#define TEST_SIMPLE_NW     1
#define TEST_LOOP          1
#define TEST_LOOP_NW       1

int a[N], b[N];
int aa[N], bb[N];

int main() {
  int i;
  int error, totError = 0;


#if TEST_SIMPLE_NW
  for (i=0; i<N; i++) a[i] = b[i] = i;

  // alloc, move to
  #pragma omp target enter data nowait map(alloc: a[0:N/4])       map(to: b[0:N/4])
  #pragma omp target enter data nowait map(alloc: a[N/4:N/4])     map(to: b[N/4:N/4])
  #pragma omp target enter data nowait map(alloc: a[N/2:N/4])     map(to: b[N/2:N/4])
  #pragma omp target enter data nowait map(alloc: a[3*(N/4):N/4]) map(to: b[3*(N/4):N/4])
  #pragma omp taskwait

  // compute
  #pragma omp target nowait map(from: a[0:N/4])       map(to: b[0:N/4])
  {
    int j;
    for(j=0; j<N/4; j++) a[j] = b[j]+1;
  }
  #pragma omp target nowait  map(from: a[N/4:N/4])     map(to: b[N/4:N/4])
  {
    int j;
    for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
  }
  #pragma omp target nowait map(from: a[N/2:N/4])     map(to: b[N/2:N/4])
  {
    int j;
    for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
  }
  #pragma omp target nowait map(from: a[3*(N/4):N/4]) map(to: b[3*(N/4):N/4])
  {
    int j;
    for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
  }
  #pragma omp taskwait


  #pragma omp target exit data nowait map(from: a[0:N/4])       map(release: b[0:N/4])
  #pragma omp target exit data nowait map(from: a[N/4:N/4])     map(release: b[N/4:N/4])
  #pragma omp target exit data nowait map(from: a[N/2:N/4])     map(release: b[N/2:N/4])
  #pragma omp target exit data nowait map(from: a[3*(N/4):N/4]) map(release: b[3*(N/4):N/4])
  #pragma omp taskwait


  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with simple nowait conpleted successfully\n");
  } else {
    printf("  test with simple nowait conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

#if TEST_LOOP
  for (i=0; i<N; i++) a[i] = b[i] = i;

  #pragma omp parallel for schedule(static, 1)
  for(i=0; i<4; i++) {
    int lb = i* N/4;
    int ub = lb + N/4;

    // alloc, move to
    #pragma omp target enter data map(alloc: a[lb:N/4]) map(to: b[lb:N/4])

    // compute
    #pragma omp target map(from: a[lb:N/4]) map(to: b[lb:N/4])
    {
      int j;
      for(j=lb; j<ub; j++) a[j] = b[j]+1;
    }

    #pragma omp target exit data map(from: a[lb:N/4]) map(release: b[lb:N/4])
  }

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with test loop wait conpleted successfully\n");
  } else {
    printf("  test with test loop wait conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

#if TEST_LOOP_NW
  for (i=0; i<N; i++) a[i] = b[i] = aa[i] = bb[i] = i;

  #pragma omp parallel for schedule(static, 1)
  for(i=0; i<4; i++) {
    int lb = i* N/4;
    int ub = lb + N/4;

    // alloc, move to
    #pragma omp target enter data nowait map(alloc: a[lb:N/4])  map(to: b[lb:N/4])
    #pragma omp target enter data nowait map(alloc: aa[lb:N/4]) map(to: bb[lb:N/4])

    // compute
    #pragma omp taskwait
    #pragma omp target nowait map(from: a[lb:N/4]) map(to: b[lb:N/4])
    {
      int j;
      for(j=lb; j<ub; j++) a[j] = b[j]+1;
    }
    #pragma omp target nowait map(from: aa[lb:N/4]) map(to: bb[lb:N/4])
    {
      int j;
      for(j=lb; j<ub; j++) aa[j] = bb[j]+1;
    }

    // get and release data
    #pragma omp taskwait
    #pragma omp target exit data nowait map(from: a[lb:N/4])  map(release: b[lb:N/4])
    #pragma omp target exit data nowait map(from: aa[lb:N/4]) map(release: bb[lb:N/4])
  }

  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: a error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  for (i=0; i<N; i++) {
    if (aa[i] != i+1) printf("%d: aa error %d != %d, error %d\n", i, aa[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with test loop nowait conpleted successfully\n");
  } else {
    printf("  test with test loop nowait conpleted with %d error(s)\n", error);
    totError++;
  }
#endif


  printf("completed with %d errors\n", totError);
  return totError;
}
