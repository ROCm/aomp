// Device-only kernel used to produce a standalone, loadable code object
// (mod.co) that carries its own instrumentation (__llvm_profile_sections +
// the device profile runtime). main.hip hipModuleLoad()s this code object, so
// in that process mod_kernel has NO host-side shadow registration. The HIP
// host-shadow drain therefore cannot see it; only the HSA-introspection drain
// (which walks every code object loaded on the agent) can collect its
// counters. That is exactly what this test asserts.
//
// When built with -DBUILD_MODULE_EXE this file is a normal HIP executable; the
// full executable link makes clang's driver add the amdgcn device profile
// runtime to the device link (addProfileRTLibs), so the embedded device code
// object is fully instrumented. run_and_check.sh extracts that object as
// mod.co. (The standalone --offload-device-only / hipcc --genco paths do NOT
// add the device profile RT, leaving __llvm_profile_instrument_gpu /
// __llvm_profile_raw_version unresolved -- hence the extract-from-executable
// approach.)

#include <hip/hip_runtime.h>

extern "C" __global__ void mod_kernel(int *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = p[i] + 7;   // taken region
  else
    p[0] = p[0] - 1;   // untaken region (non-trivial coverage)
}

#ifdef BUILD_MODULE_EXE
int main() {
  const int n = 64;
  int *d = nullptr;
  if (hipMalloc(&d, n * sizeof(int)) != hipSuccess)
    return 1;
  if (hipMemset(d, 0, n * sizeof(int)) != hipSuccess)
    return 2;
  mod_kernel<<<dim3(1), dim3(64)>>>(d, n);
  if (hipDeviceSynchronize() != hipSuccess)
    return 3;
  (void)hipFree(d);
  return 0;
}
#endif
