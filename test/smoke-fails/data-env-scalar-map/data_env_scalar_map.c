#include <stdio.h>

#include <omp.h>

int
main(int argc, char *argv[], char **envp)
{
  int   numdev = omp_get_num_devices();

  printf ("Machine has %d GPU device%s\n", numdev, (numdev==1 ? "" : "s") );

  int from = 13;
  int tofrom = 17;

  printf("ON HOST before: from = %d, tofrom = %d\n", from, tofrom);

  #pragma omp target data map(from:from) map(tofrom:tofrom)
  #pragma omp target
  {
    printf("ON GPU: enter from = %d, tofrom = %d\n", from, tofrom); 

    from = 5;
    tofrom = 5; 

    printf("ON GPU: exit from = %d, tofrom = %d\n", from, tofrom); 
  }

  // This should print ON HOST after: from = 5, tofrom = 5
  printf("ON HOST after: from = %d, tofrom = %d\n", from, tofrom);
  if (from != 5 || tofrom != 5) {printf("failed\n"); return -1;}
  return 0;
}

