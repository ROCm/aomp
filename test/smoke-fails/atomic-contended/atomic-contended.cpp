#include <omp.h>
#include <stdio.h>

// Note: There are four scenarios where the AMDGPU atomic optimizer will be
//       applicable (w.r.t. contended atomics): (1) contended atomic with
//       constant stride (2) contended atomic with variable stride and (3) &
//       (4) are variants where the resulting sum is written back as 'result'.
//       The following testcase will implement scenario (2), by using the
//       thread-specific iteration as variable stride.
void atomicAddUniform(int *Index, int *Step, int N) {
#pragma omp target teams distribute parallel for map(tofrom : *Index)
  for (int i = 0; i < N; ++i) {
#pragma omp atomic
    *Index += i;
  }
}

int main() {
  int Contended = 0;
  int N = 4096;
  int Step[N];
  // Expected result is the sum of all N integers.
  int Expected = ((N-1)*(N)) >> 1;

  atomicAddUniform(&Contended, Step, N);

  int Err = (Contended == Expected) ? 0 : 1;

  if (!Err)
    printf("Success\n");
  else
    printf("Fail\n");

  return Err;
}

// Note: All prefixes will verify the presence of certain 'unique' instructions
//       as well as the absence of instructions emitted by other optimizations.

/// DPP-NOT: v_readlane_b32
/// DPP-NOT: s_lshl_b64
/// DPP-NOT: s_add_i32
/// DPP: v_mbcnt_lo_u32_b32
/// DPP: v_mbcnt_hi_u32_b32
/// DPP: v_add_u32_dpp {{.*}} row_shr:1
/// DPP: v_add_u32_dpp {{.*}} row_shr:2
/// DPP: v_add_u32_dpp {{.*}} row_shr:4
/// DPP: v_add_u32_dpp {{.*}} row_shr:8
/// DPP: v_add_u32_dpp {{.*}} row_bcast:15
/// DPP: v_add_u32_dpp {{.*}} row_bcast:31
/// DPP: global_atomic_add

/// ITERATIVE-NOT: v_add_u32_dpp
/// ITERATIVE: v_readlane_b32
/// ITERATIVE: s_lshl_b64
/// ITERATIVE: s_add_i32
/// ITERATIVE: v_mbcnt_lo_u32_b32
/// ITERATIVE: v_mbcnt_hi_u32_b32
/// ITERATIVE-NOT: v_add_u32_dpp
/// ITERATIVE: global_atomic_add

/// NONE-NOT: v_readlane_b32
/// NONE-NOT: s_lshl_b64
/// NONE-NOT: s_add_i32
/// NONE-NOT: v_mbcnt_lo_u32_b32
/// NONE-NOT: v_mbcnt_hi_u32_b32
/// NONE-NOT: v_add_u32_dpp
/// NONE: global_atomic_add
