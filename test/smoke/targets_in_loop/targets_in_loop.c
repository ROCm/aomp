#include <stdio.h>
#include <omp.h>

int main(void) {

#ifndef KLIMIT
#define KLIMIT 100
#endif
for (int k=0; k<KLIMIT ; k++) {
  printf(" in host loop interation %d\n",k);
  #pragma omp target
  printf(" on device iteration %d\n",k);

}

printf(" loop complete \n");
return 0;
}

