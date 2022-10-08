#include <stdio.h>
#include <omp.h>

int main()
{
  int N = 1000;
  
  int b[N];
  unsigned c[N];

  for (int i=0; i<N; i++) {
    b[i] = i+1;
    c[i] = i+2;
  }

  int8_t int8_sum = 0;
  int16_t int16_sum = 0;
  int32_t int32_sum = 0;
  uint32_t uint32_sum = 0;
  int64_t int64_sum = 0;
  uint64_t uint64_sum = 0;
  
#pragma omp target teams distribute parallel for map(tofrom:int8_sum) reduction(+:int8_sum)
  for (int j = 0; j< 10; j=j+1)
    int8_sum += b[j];

#pragma omp target teams distribute parallel for map(tofrom:int16_sum) reduction(+:int16_sum)
  for (int j = 0; j< 100; j=j+1)
    int16_sum += c[j];

#pragma omp target teams distribute parallel for reduction(+:int32_sum)
  for (int j = 0; j< N; j=j+1)
    int32_sum += b[j] + c[j];

#pragma omp target teams distribute parallel for map(tofrom:uint32_sum) reduction(+:uint32_sum)
  for (int j = 0; j< N; j=j+1)
    uint32_sum += b[j] + c[j];

#pragma omp target teams distribute parallel for map(tofrom:int64_sum) reduction(+:int64_sum)
  for (int j = 0; j< N; j=j+1)
    int64_sum += b[j] + c[j];

#pragma omp target teams distribute parallel for map(tofrom:uint64_sum) reduction(+:uint64_sum)
  for (int j = 0; j< N; j=j+1)
    uint64_sum += b[j] + c[j];

  printf("%d %d %d %u %ld %lu\n", int8_sum, int16_sum, int32_sum, uint32_sum, int64_sum, uint64_sum);
  
  int rc = (int8_sum != 55) || (int16_sum != 5150) || (int32_sum != 1002000) ||
    (uint32_sum != 1002000) || (int64_sum != 1002000) || (uint64_sum != 1002000);

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:2
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8
/// CHECK: DEVID:[[S:[ ]*]][[DEVID:[0-9]+]] SGN:8


