#include <omp.h>
#include <stdio.h>

#pragma omp requires unified_shared_memory

void step_physics_usm(int n, float* pos, float* vel, float dt, int* ran_on_device) {
int device_flag = 0;

#pragma omp target map(tofrom: device_flag)
{
if (!omp_is_initial_device()) {
device_flag = 1;
}
}

#pragma omp target teams distribute parallel for //simd
for(int i = 0; i < n; i++) {
pos[i] += vel[i] * dt;
}

*ran_on_device = device_flag;
}

int n = 100000;

int main()
{
float pos[n];
float vel[n];
float dt = 0.016;

for (int i = 0; i < n; i++) {
pos[i] = 0.0;
vel[i] = 10.0;
}
int ranOnDevice = 0;

printf("Sending computation to OpenMP...");

// Pass the raw memory pointers to the C function
step_physics_usm(n, &pos[0], &vel[0], dt, &ranOnDevice);

// Verify the results
if (ranOnDevice == 1)
printf("[SUCCESS] Computation successfully offloaded to the APU!");
else {
printf("[WARNING] Computation fell back to the Host CPU.\n");
printf(" Check your compiler offload flags, OpenMP installation, and GPU drivers.\n");
}

printf("Sample result: pos[0] = %f\n", pos[0]);

return 0;
}
