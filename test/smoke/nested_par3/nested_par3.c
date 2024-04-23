#include <stdio.h>
#define N   5


int main (void)
{
  long int aa=0;
  int res = 0;

  int ng =6;
  int cmom = 4;
  int nxyz = 5;
#pragma omp target teams distribute num_teams(nxyz) thread_limit(4) map(tofrom:aa)
  for (int gid = 0; gid < nxyz; gid++) {
    #pragma omp parallel for  collapse(2)
    for (unsigned int g = 0; g < ng; g++) {
      for (unsigned int l = 0; l < cmom-1; l++) {
        int a = 0;
        for (int ii = 0; ii < N+2; ii++) {
	  #pragma omp parallel for reduction(+:a)
          for (int i = 0; i < N; i++) {
            a += i;
          }
        }
        #pragma omp atomic
        aa += a;
      }
    }
  }
  long exp = (long)ng*(cmom-1)*nxyz*(N*(N-1)/2)*(N+2);
  fprintf (stderr, "The result is = %ld exp:%ld!\n", aa,exp);
  if (aa != exp) {
    fprintf(stderr, "Failed %ld\n",aa);
    return 1;
  }
  aa = 0;
#if 0
  #pragma omp target teams distribute num_teams(nxyz) thread_limit(4) map(tofrom:aa)
  for (int gid = 0; gid < nxyz; gid++) {
    for (unsigned int g = 0; g < ng; g++) {
      for (unsigned int l = 0; l < cmom-1; l++) {
        int a = 0;
        for (int ii = 0; ii < N+2; ii++) {
          #pragma omp parallel for reduction(+:a)
          for (int i = 0; i < N; i++) {
            a += i;
          }
        }
        #pragma omp atomic
        aa += a;
      }
    }
  }
  exp = (long)ng*(cmom-1)*nxyz*(N*(N-1)/2)*(N+2);
  fprintf (stderr, "The result is = %ld exp:%ld!\n", aa,exp);
  if (aa != exp) {
    fprintf(stderr, "Failed %ld\n",aa);
    return 1;
  }
  aa = 0;
#pragma omp target teams distribute num_teams(nxyz) thread_limit(4) map(tofrom:aa)
  for (int gid = 0; gid < nxyz; gid++) {
    #pragma omp parallel for
    for (unsigned int g = 0; g < ng; g++) {
      #pragma omp parallel for
      for (unsigned int l = 0; l < cmom-1; l++) {
        int a = 0;
        #pragma omp parallel for
        for (int ii = 0; ii < N+2; ii++) {
          #pragma omp parallel for reduction(+:a)
          for (int i = 0; i < N; i++) {
            a += i;
          }
        }
        #pragma omp atomic
        aa += a;
      }
    }
  }
  exp = (long)ng*(cmom-1)*nxyz*(N*(N-1)/2)*(N+2);
  fprintf (stderr, "The result is = %ld exp:%ld!\n", aa,exp);
  if (aa != exp) {
    fprintf(stderr, "Failed %ld\n",aa);
    return 1;
  }
  aa = 0;
#pragma omp target teams distribute num_teams(nxyz) thread_limit(7) map(tofrom:aa)
  for (int gid = 0; gid < nxyz; gid++) {
    #pragma omp parallel for  collapse(2)
    for (unsigned int g = 0; g < ng; g++) {
      for (unsigned int l = 0; l < cmom-1; l++) {
        int a = 0;
        #pragma omp parallel for
        for (int ii = 0; ii < N+2; ii++) {
          #pragma omp parallel for reduction(+:a)
          for (int i = 0; i < N; i++) {
            a += i;
          }
        }
        #pragma omp atomic
        aa += a;
      }
    }
  }
  fprintf (stderr, "The result is = %ld exp:%ld!\n", aa,exp);
  if (aa != exp) {
    fprintf(stderr, "Failed %ld\n",aa);
    return 1;
  }
  #endif
  return 0;
}
