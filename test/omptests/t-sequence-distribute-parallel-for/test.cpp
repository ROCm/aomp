#define lnv (512)

#include <stdio.h>
#include <stdlib.h>

int value[lnv];
  
int main(int argc, char *argv[]) {
  int flag1 =1;
  
  for (auto &a : value)
    a = 0;

  #pragma omp target data map(tofrom: value) 
  {
    #pragma omp target 
    #pragma omp teams num_teams(8)
    {
      int i;

      if (flag1 == 1) {
        #pragma omp distribute parallel for 
        for(i=0; i < lnv; i++)
        {
          value[i] += 1;
        }
        
        #pragma omp distribute parallel for 
        for(i=0; i < lnv; i++) {
          value[i] += 1;
        }
      }
    }
  }
  
  for (auto &a : value)
    printf("%d\n",a);
  return 0;
}
