#include <stdio.h>
#include <omp.h>
int main(void) {
  bool a = true;
  bool b = true;
  bool target_is_on_host = false;
  #pragma omp target if(a && b) map(from:target_is_on_host)
  {
     target_is_on_host = omp_is_initial_device();
  }
  if (target_is_on_host) return 1; // it should be on device
				  
  // 18.0-0 fails in codegen with if clause in target data but not target 
  bool target_data_is_on_device = true ;
  #pragma omp target data if(a && b) map(from:target_data_is_on_device)
  {
     target_data_is_on_device = !omp_is_initial_device();
  }
  if (target_data_is_on_device) return 1; // it should be on host
  return 0;
}
