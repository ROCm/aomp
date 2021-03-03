#include <stdio.h>
#include <omp.h>

int arr[100];
int nt =12;
int main()
{
  fprintf(stderr, "Omp host get_num_devices %d\n", omp_get_num_devices());
#pragma omp target teams distribute parallel for num_threads(nt)
  for (int i=0; i<100;i++)
    arr[i] =i;

//Verify
  int errors = 0;
  for (int i=0; i<100;i++){
    if(arr[i] != i)
			errors++;
}
  if(!errors){
    fprintf(stderr, "Success\n");
    return 0;
  } else{
    fprintf(stderr, "Failed\nErrors: %d\n", errors);
    return 1;
  }
}

