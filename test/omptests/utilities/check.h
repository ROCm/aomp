#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int offloading_disabled()
{
  char *xlStr=  getenv("OMP_TARGET_OFFLOAD");
  if (! xlStr) return 0;
  if (0==strcmp(xlStr, "DISABLED")) return 1;  
  return 0;
}

int check_offloading(){

  int A[1] = {-1};

  if (offloading_disabled())
    A[0] = 0;
  else 
  {
    #pragma omp target
    {
      A[0] = omp_is_initial_device();
    }
  }

  if (!A[0]){
    printf("Able to use offloading!\n");
    return 0;
  }
  else
    printf("### Unable to use offloading!  8^( ###\n");
  

  return 1;
}
