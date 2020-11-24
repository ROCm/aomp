#include <omp.h>
#include <stdio.h>

int main() {

  const int num_devices = omp_get_num_devices();

  if (num_devices > 0) {
    for (unsigned i = 0; i < (unsigned)num_devices; i++) {
      int num;
#pragma omp target map(from : num) device(i)
      num = omp_get_device_num();

      if (num != i) {
        printf("Fail: Device %u returned id %u\n", i, num);
        return 1;
      }
    }
  }

  return 0;
}
