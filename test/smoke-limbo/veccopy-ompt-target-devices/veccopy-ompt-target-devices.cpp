#include <stdio.h>
#include <cassert>
#include <omp.h>

#include "callbacks.h"

// Map of devices traced
DeviceMap_t DeviceMap;

int main()
{
  int N = 100000;

  int a[N];
  int b[N];

  int i;

  for (i=0; i<N; i++)
    a[i]=0;

  for (i=0; i<N; i++)
    b[i]=i;

  // Warm up
#pragma omp target 
  {
  }

  for (auto Itr : DeviceMap)
    flush_trace(Itr.second);
  
  for (int dev = 0; dev < omp_get_num_devices(); ++dev) {
#pragma omp target teams distribute parallel for device(dev)
    {
      for (int j = 0; j< N; j++)
	a[j]=b[j];
    }
  }

  for (auto Itr : DeviceMap) {
    flush_trace(Itr.second);
    stop_trace(Itr.second);
  }
  
  int rc = 0;
  for (i=0; i<N; i++)
    if (a[i] != b[i] ) {
      rc++;
      printf ("Wrong value: a[%d]=%d\n", i, a[i]);
    }

  if (!rc)
    printf("Success\n");

  return rc;
}

/// CHECK: Record Submit

