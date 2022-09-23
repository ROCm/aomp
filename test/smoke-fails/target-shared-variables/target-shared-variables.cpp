#include <stdio.h>
#include <iostream>


void further_update_function(float x, float y, float z, float *px, float *py, float *pz) {
  float local_x = 1.0;
  float local_y = 2.0;
  float local_z = 3.0;
  for (int i = 0; i < (int)x; ++i) {
    local_x += x * x + i;
    local_y += y * y + i;
    local_z += z * z + i;
  }

  *px = local_x;
  *py = local_y;
  *pz = local_z;
}

void update_function_no_LDS(float x, float y, float z, float *output) {
  float local_x = 0.0;
  float local_y = 0.0;
  float local_z = 0.0;
  further_update_function(x, y, z, &local_x, &local_y, &local_z);

  *output = local_x / (local_x + 1);
  *output += local_y / (local_x + 1);
  *output += local_z / (local_x + 1);
}

void update_function_uses_LDS(float x, float y, float z, float *output) {
  float local_x = 0.0;
  float local_y = 0.0;
  float local_z = 0.0;
  further_update_function(x, y, z, &local_x, &local_y, &local_z);

  float rx = 0.0;
  float ry = 0.0;
  float rz = 0.0;
  further_update_function(x, y, z, &rx, &ry, &rz);

  *output = local_x / (rx + 1);
  *output += local_y / (ry + 1);
  *output += local_z / (rz + 1);
}

int main () {
  int size = 20000;
  float arr[size];

  for (int i=0; i<size; ++i) {
    arr[i] = i;
  }

  #pragma omp target data map(tofrom:arr[0:size])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < size; ++i) {
    float half = arr[i] * 0.5;
    float output = 0.0;

    update_function_no_LDS(half - 1.0, half + 1.0, half, &output);

    arr[i] = output;
  }

  #pragma omp target data map(tofrom:arr[0:size])
  #pragma omp target teams distribute parallel for
  for (int i = 0; i < size; ++i) {
    float half = arr[i] * 0.5;
    float output = 0.0;

    update_function_uses_LDS(half - 1.0, half + 1.0, half, &output);

    arr[i] = output;
  }

  if (arr[7] - 5.580645 <= 0.0001)
    printf("SUCCESS\n");
  else
    printf("FAIL\n");
}
