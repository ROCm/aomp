#!/bin/bash
# OmpDevCov check driver: device-side PGO round trip for OpenMP offload.
#
# Args (passed from the Makefile):
#   $1 = AOMP       (LLVM dir containing bin/clang, llvm-profdata)
#   $2 = GPU arch   (e.g. gfx90a, may include :features)
#   $3 = FILECHECK  (path to FileCheck)
#
# omp_pgo (the -fprofile-generate binary) is built by the harness. Here we:
#   1. run it -> host default.profraw + device amdgcn-amd-amdhsa.*.profraw
#      (the device file is produced ONLY by the HSA drain),
#   2. assert a device profraw was produced and merge,
#   3. FileCheck the merged profile for the device offload function,
#   4. rebuild with -fprofile-use and assert it consumes cleanly (no profile
#      mismatch / out-of-date diagnostics) and runs.

set -u
AOMP="$1"; ARCH="$2"; FILECHECK="$3"
CLANG="$AOMP/bin/clang"
PROFDATA="$AOMP/bin/llvm-profdata"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here" || exit 1

# Capability gate: skip cleanly if the toolchain has no amdgcn device profile
# runtime (i.e. no device PGO / HSA drain yet), so CI is not red before the
# feature lands.
resdir="$("$CLANG" -print-resource-dir 2>/dev/null)"
if [ ! -f "$resdir/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a" ]; then
  echo "SKIP OmpDevCov: no amdgcn device profile runtime in toolchain"
  exit 0
fi

set -e
rm -f ./*.profraw merged.profdata omp_pgo_use 2>/dev/null || true

run_env() { LD_LIBRARY_PATH="$AOMP/lib:${LD_LIBRARY_PATH:-}" "$@"; }

# 1. Run the generate binary (built by the harness as ./omp_pgo). Build it here
#    too if missing, so the script also works standalone.
if [ ! -x ./omp_pgo ]; then
  "$CLANG" -fopenmp --offload-arch="$ARCH" -fprofile-generate \
    omp_pgo.c -o omp_pgo -Wl,-rpath,"$AOMP/lib"
fi
run_env ./omp_pgo

# 2. The HSA drain must have produced at least one device profraw.
shopt -s nullglob
dev=(amdgcn-amd-amdhsa*.profraw)
if [ ${#dev[@]} -eq 0 ]; then
  echo "FAIL OmpDevCov: no device (amdgcn-amd-amdhsa) profraw -- HSA drain did not fire"
  ls -1 ./*.profraw 2>/dev/null || true
  exit 1
fi
echo "OmpDevCov: device profraw(s): ${dev[*]}"

"$PROFDATA" merge -o merged.profdata ./*.profraw

# 3. The merged profile must carry the device offload function counters.
"$PROFDATA" show --all-functions merged.profdata | "$FILECHECK" check.txt

# 4. -fprofile-use round trip: must consume the profile without a mismatch /
#    out-of-date diagnostic, then run.
use_log="$(mktemp)"
"$CLANG" -fopenmp --offload-arch="$ARCH" -fprofile-use=merged.profdata \
  omp_pgo.c -o omp_pgo_use -Wl,-rpath,"$AOMP/lib" 2> "$use_log" || {
    echo "FAIL OmpDevCov: -fprofile-use build failed"; cat "$use_log"; rm -f "$use_log"; exit 1; }
if grep -Eiq "out of date|profile data may be out of date|no profile data available|mismatch" "$use_log"; then
  echo "FAIL OmpDevCov: -fprofile-use reported stale/mismatched profile"
  cat "$use_log"; rm -f "$use_log"; exit 1
fi
rm -f "$use_log"

run_env ./omp_pgo_use

echo "OmpDevCov PASSED"
