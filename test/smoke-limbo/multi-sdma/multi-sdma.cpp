
#define N 1024

int main(int argc, char **argv) {

  int *pA = new int[N];
  int *pB = new int[N];
  double *pD = new double[N];

#pragma omp target teams distribute parallel for map(from: pA[0:N])
  for (int i = 0; i < N; ++i)
    pA[i] = 1;

#pragma omp target teams distribute parallel for map(from: pB[0:N])
  for (int i = 0; i < N; ++i)
    pB[i] = 2;

#pragma omp target teams distribute parallel for map(tofrom: pA[0:N], pB[0:N]) map(from: pD[0:N])
  for (int i = 0; i < N; ++i) {
    pD[i] = pA[i] * pB[i];
  }

  if (pD[2] != 2)
    return 7;

  return 0;
}

// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 1
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 2
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 1
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 2
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 1
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 2
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 1
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 2
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 1
// CHECK: TARGET AMDGPU RTL --> Running Async Copy on SDMA Engine: 2
