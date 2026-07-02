// HipDevCov: prove the HSA-introspection device-coverage drain captures device
// code that the HIP host-shadow drain cannot see.
//
// This program contains exactly one statically-registered kernel (host_kernel)
// -- it has a host-side shadow, so the host-shadow drain covers it. It then
// hipModuleLoad()s mod.co (built+extracted by run_and_check.sh) and launches
// mod_kernel, which is NOT compiled into this program and therefore has no host
// shadow here. mod_kernel's device counters can only reach the profile via the
// HSA drain. run_and_check.sh asserts BOTH kernels appear in the merged
// profile, so a passing run requires the HSA drain to be working.

#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void host_kernel(int *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = p[i] * 3;   // taken region
  else
    p[0] = p[0] - 1;   // untaken region
}

int main() {
  const int n = 64;
  int *d = nullptr;
  if (hipMalloc(&d, n * sizeof(int)) != hipSuccess)
    return 1;
  if (hipMemset(d, 0, n * sizeof(int)) != hipSuccess)
    return 2;

  // (1) host-shadow path: a normally-registered kernel.
  host_kernel<<<dim3(1), dim3(64)>>>(d, n);
  if (hipDeviceSynchronize() != hipSuccess)
    return 3;

  // (2) HSA-only path: a kernel from a separately loaded code object, with no
  // host shadow in this process.
  hipModule_t mod;
  if (hipModuleLoad(&mod, "mod.co") != hipSuccess) {
    fprintf(stderr, "hipModuleLoad(mod.co) failed\n");
    return 4;
  }
  hipFunction_t fn;
  if (hipModuleGetFunction(&fn, mod, "mod_kernel") != hipSuccess) {
    fprintf(stderr, "hipModuleGetFunction(mod_kernel) failed\n");
    return 5;
  }
  int nn = n;
  void *args[] = {&d, &nn};
  if (hipModuleLaunchKernel(fn, 1, 1, 1, 64, 1, 1, 0, nullptr, args, nullptr) !=
      hipSuccess)
    return 6;
  if (hipDeviceSynchronize() != hipSuccess)
    return 7;

  // Intentionally do NOT hipModuleUnload(mod): the HSA-introspection drain runs
  // at process exit and walks the code objects still loaded on the agent.
  // Unloading mod.co here would remove its counters before the drain sees them.
  (void)hipFree(d);
  printf("HipDevCov: ran host_kernel + module mod_kernel\n");
  return 0;
}
