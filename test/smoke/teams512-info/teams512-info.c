#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 131072;

  int a[N]; int b[N]; int w[N]; int t[N]; int s[N];
  for (int i=0; i<N; i++) {
    a[i]=0;
    b[i]=i;
  }

#pragma omp target teams distribute parallel for num_teams(N/256) 
    for (int j = 0; j< N; j++) {
      a[j]=b[j];
#ifdef GPUINFO
      t[j] = omp_get_team_num();
      w[j] = omp_ext_get_warp_id();
      s[j] = omp_ext_get_smid();
#endif
  }
  int rc = 0;
  int lastW = -1;
  printf("X  thread team  CUid\n");
  for (int i=0; i<N; i++) {
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong varlue: a[%d]=%d\n", i, a[i]);
    }
#ifdef GPUINFO
    if (t[i] != lastW) {
//  if (w[i] != lastW) {
      printf("X %6d %4d %5d\n", i, t[i], s[i]);
//    lastW = w[i];
      lastW = t[i];
    }
#endif
  }
  if (!rc)
    printf("Success\n");

  return rc;
}
