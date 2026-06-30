#!/bin/bash
# HipDevCov check driver.
#
# Args (passed from the Makefile):
#   $1 = AOMP        (LLVM dir containing bin/clang, llvm-profdata, llvm-objdump)
#   $2 = GPU arch    (e.g. gfx90a, may include :features)
#   $3 = ROCM_PATH   (HIP/ROCm install for --rocm-path and libamdhip64)
#
# Proves the HSA-introspection drain captured device coverage that the HIP
# host-shadow drain cannot see. Verifies, against the actual merged profile:
#   * the HSA drain produced its own profraw (uniquely named *.hsa<N>.*.profraw),
#   * mod_kernel (loaded via hipModuleLoad, no host shadow) is present with a
#     NONZERO function count -- only possible via the HSA drain,
#   * host_kernel (host-shadow path) is also present with a nonzero count.
# The human-readable profile dump is left in devcov-profile.txt for inspection.

set -u
AOMP="$1"; ARCH="$2"; ROCM="$3"
CLANG="$AOMP/bin/clang"
PROFDATA="$AOMP/bin/llvm-profdata"
OBJDUMP="$AOMP/bin/llvm-objdump"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here" || exit 1

fail() { echo "FAIL HipDevCov: $*"; exit 1; }

# Function count for a front-end-instrumented function in the --counts dump.
func_count() {
  awk -v fn="$1" '
    $0 ~ ("^[ \t]*" fn ":$") { inblk = 1; next }
    inblk && /Function count:/ { print $3; exit }
  ' "$2"
}

# Capability gate: skip cleanly only when the toolchain truly cannot do device
# PGO (no amdgcn device profile runtime / no host HSA-drain runtime). Leaves a
# marker so the skip is visible in archived results rather than a silent pass.
resdir="$("$CLANG" -print-resource-dir 2>/dev/null)"
devrt="$resdir/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"
rm -f DEVCOV-SKIPPED.txt
if [ ! -f "$devrt" ] || ! ls "$resdir"/lib/*/libclang_rt.profile_rocm*.a >/dev/null 2>&1; then
  echo "no device profile runtime / HSA-drain runtime in $resdir" > DEVCOV-SKIPPED.txt
  echo "SKIP HipDevCov: $(cat DEVCOV-SKIPPED.txt)"
  exit 0
fi

set -e
rm -f ./*.profraw mod.co merged.profdata devcov-profile.txt builder \
      builder.*.host-* builder.*.hip-amdgcn-amd-amdhsa--* 2>/dev/null || true

# 1. Build mod.hip as a full executable so the device link gets the device
#    profile RT, then extract its device code object as the loadable mod.co.
"$CLANG" -x hip --offload-arch="$ARCH" -fno-gpu-rdc -DBUILD_MODULE_EXE \
  -fprofile-instr-generate -fcoverage-mapping --rocm-path="$ROCM" \
  mod.hip -o builder -L"$ROCM/lib" -lamdhip64 -Wl,-rpath,"$ROCM/lib"
"$OBJDUMP" --offloading builder >/dev/null 2>&1 || true
shopt -s nullglob
extracted=(builder*.hip-amdgcn-amd-amdhsa--*gfx*)
[ ${#extracted[@]} -gt 0 ] || fail "could not extract device code object from builder"
cp "${extracted[0]}" mod.co

# 2. main is built by the harness (TESTNAME=main); build it here if missing.
if [ ! -x ./main ]; then
  "$CLANG" -x hip --offload-arch="$ARCH" -fno-gpu-rdc \
    -fprofile-instr-generate -fcoverage-mapping --rocm-path="$ROCM" \
    main.hip -o main -L"$ROCM/lib" -lamdhip64 -Wl,-rpath,"$ROCM/lib"
fi

# 3. Run with mod.co in CWD.
rm -f ./*.profraw
LLVM_PROFILE_FILE="$here/host.profraw" \
  LD_LIBRARY_PATH="$ROCM/lib:${LD_LIBRARY_PATH:-}" \
  ./main

# 4. The HSA-introspection drain writes its own, uniquely-named profraw
#    (<arch>.hsa<N>.<file>.profraw). Its presence is a direct fingerprint that
#    the HSA pass ran (the host-shadow drain names files <arch>[:feat].<file>).
hsa=(*.hsa[0-9]*.profraw)
[ ${#hsa[@]} -gt 0 ] || fail "no HSA-drain profraw (*.hsa<N>.*.profraw) produced"
echo "HipDevCov: HSA-drain profraw(s): ${hsa[*]}"

# 5. Merge and verify real counts from the dump.
"$PROFDATA" merge -sparse -o merged.profdata ./*.profraw
"$PROFDATA" show --all-functions --counts merged.profdata > devcov-profile.txt

mc="$(func_count mod_kernel devcov-profile.txt)"
hc="$(func_count _Z11host_kernelPii devcov-profile.txt)"
[ -n "$mc" ] || fail "mod_kernel absent from merged profile (HSA drain did not capture it)"
[ -n "$hc" ] || fail "host_kernel absent from merged profile"
[ "$mc" -gt 0 ] 2>/dev/null || fail "mod_kernel function count is '$mc' (expected > 0)"
[ "$hc" -gt 0 ] 2>/dev/null || fail "host_kernel function count is '$hc' (expected > 0)"

echo "HipDevCov PASSED (mod_kernel=$mc via HSA drain, host_kernel=$hc)"
