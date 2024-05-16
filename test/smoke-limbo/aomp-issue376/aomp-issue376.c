/* Based on https://github.com/ROCm-Developer-Tools/aomp/issues/376 */

#include <stdio.h>
#include <assert.h>
#include <omp.h>

int main()
{
  int initial_device=1;

#pragma omp target data map(tofrom : initial_device)
  {
      int a;
      #pragma omp target
      {
          for(int i=0; i<100; i++){
          a+=i;
        }
      }
  }
#pragma omp target data map(tofrom: initial_device) device(0)
  {
      int a;
      #pragma omp target
      {
          for(int i=0; i<100; i++){
          a+=i;
        }
      }
  }

  return 0;
}

/// CHECK: Callback Target
/// CHECK-SAME: device_num=0





