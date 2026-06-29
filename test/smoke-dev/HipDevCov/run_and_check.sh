#!/bin/bash
# HipDevCov check driver.
#
# Args (passed from the Makefile):
#   $1 = AOMP        (LLVM dir containing bin/clang, llvm-profdata, llvm-objdump)
#   $2 = GPU arch    (e.g. gfx90a, may include :features)
#   $3 = FILECHECK   (path to FileCheck)
#   $4 = ROCM_PATH   (HIP/ROCm install for --rocm-path and libamdhip64)
#
# Returns 0 only if BOTH the host-shadow kernel and the module-only kernel are
# present in the merged profile (the latter requires the HSA drain).

set -u
AOMP="$1"; ARCH="$2"; FILECHECK="$3"; ROCM="$4"
CLANG="$AOMP/bin/clang"
PROFDATA="$AOMP/bin/llvm-profdata"
OBJDUMP="$AOMP/bin/llvm-objdump"
COV="$AOMP/bin/llvm-cov"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here" || exit 1

# Capability gate: skip cleanly on toolchains that do not ship the device
# profile runtime / HSA drain yet (so this does not red-fail CI before the
# feature lands). A real, drain-capable toolchain has the amdgcn device profile
# RT and the host drain symbol.
resdir="$("$CLANG" -print-resource-dir 2>/dev/null)"
devrt="$resdir/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"
if [ ! -f "$devrt" ]; then
  echo "SKIP HipDevCov: no device profile runtime at $devrt"
  exit 0
fi
if ! ls "$resdir"/lib/*/libclang_rt.profile_rocm*.a >/dev/null 2>&1; then
  echo "SKIP HipDevCov: no host profile_rocm runtime (HSA drain) in toolchain"
  exit 0
fi

set -e
rm -f ./*.profraw mod.co merged.profdata builder builder.*.host-* \
      builder.*.hip-amdgcn-amd-amdhsa--* 2>/dev/null || true

# 1. Build mod.hip as a full executable so the device link gets the device
#    profile RT, then extract its device code object as the loadable mod.co.
"$CLANG" -x hip --offload-arch="$ARCH" -fno-gpu-rdc -DBUILD_MODULE_EXE \
  -fprofile-instr-generate -fcoverage-mapping --rocm-path="$ROCM" \
  mod.hip -o builder -L"$ROCM/lib" -lamdhip64 -Wl,-rpath,"$ROCM/lib"
"$OBJDUMP" --offloading builder >/dev/null 2>&1 || true
co="$(ls builder*.hip-amdgcn-amd-amdhsa--*gfx* 2>/dev/null | head -1)"
if [ -z "$co" ]; then
  echo "FAIL HipDevCov: could not extract device code object from builder"
  exit 1
fi
cp "$co" mod.co

# 2. main is built by the smoke harness (TESTNAME=main). Build it here too if it
#    is missing, so the script also works when run standalone.
if [ ! -x ./main ]; then
  "$CLANG" -x hip --offload-arch="$ARCH" -fno-gpu-rdc \
    -fprofile-instr-generate -fcoverage-mapping --rocm-path="$ROCM" \
    main.hip -o main -L"$ROCM/lib" -lamdhip64 -Wl,-rpath,"$ROCM/lib"
fi

# 3. Run. Device .profraw files are written to CWD (arch-prefixed for the
#    host-shadow drain, arch.hsa<N>-prefixed for the HSA drain); the host one
#    goes to LLVM_PROFILE_FILE.
rm -f ./*.profraw
LLVM_PROFILE_FILE="$here/host.profraw" \
  LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}" \
  ./main

# 4. Merge host + all device profraws and assert both kernels are present.
"$PROFDATA" merge -sparse -o merged.profdata ./*.profraw
"$PROFDATA" show --all-functions merged.profdata | "$FILECHECK" check.txt

# 5. Sanity (non-fatal): llvm-cov can consume the merged device+host profile.
#    main's covmap only describes host_kernel/main, so llvm-cov warns about the
#    module-only function; that is expected, hence non-fatal.
"$COV" report ./main -instr-profile=merged.profdata >/dev/null 2>&1 || true

echo "HipDevCov PASSED"
