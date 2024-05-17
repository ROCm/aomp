#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

int main(){
    #pragma omp target
	{
#pragma omp error at (execution) severity(warning) message("im an error")
      }
    printf("Done\n");
    return 0;
}

