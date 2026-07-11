#!/bin/bash
# OmpDevCov check driver: device-side PGO round trip for OpenMP offload.
#
# Args (passed from the Makefile):
#   $1 = AOMP       (LLVM dir containing bin/clang, llvm-profdata)
#   $2 = GPU arch   (e.g. gfx90a, may include :features)
#
# OpenMP offload has no host-shadow drain, so device counters can only reach the
# profile through the HSA-introspection drain. omp_pgo (the -fprofile-generate
# binary) is built by the harness. Here we verify, against the actual merged
# profile, that:
#   * the HSA drain produced a device profraw (amdgcn-amd-amdhsa.*.profraw),
#   * the device function classify has NONZERO counts on BOTH branches of its
#     data-dependent if/else -- real device PGO data, not just a present symbol,
#   * -fprofile-use consumes the merged profile cleanly and runs.
# The human-readable profile dump is left in devcov-profile.txt for inspection.

set -u
AOMP="$1"; ARCH="$2"
CLANG="$AOMP/bin/clang"
PROFDATA="$AOMP/bin/llvm-profdata"

here="$(cd "$(dirname "$0")" && pwd)"
cd "$here" || exit 1

fail() { echo "FAIL OmpDevCov: $*"; exit 1; }
run_env() { LD_LIBRARY_PATH="$AOMP/lib:${LD_LIBRARY_PATH:-}" "$@"; }

# Capability gate: skip cleanly only when there is no amdgcn device profile
# runtime (no device PGO possible). Leave a marker so the skip is visible.
resdir="$("$CLANG" -print-resource-dir 2>/dev/null)"
rm -f DEVCOV-SKIPPED.txt
if [ ! -f "$resdir/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a" ]; then
  echo "no amdgcn device profile runtime in $resdir" > DEVCOV-SKIPPED.txt
  echo "SKIP OmpDevCov: $(cat DEVCOV-SKIPPED.txt)"
  exit 0
fi

set -e
rm -f ./*.profraw merged.profdata devcov-profile.txt omp_pgo_use 2>/dev/null || true

# 1. Run the generate binary (built by the harness as ./omp_pgo).
if [ ! -x ./omp_pgo ]; then
  "$CLANG" -fopenmp --offload-arch="$ARCH" -fprofile-generate \
    omp_pgo.c -o omp_pgo -Wl,-rpath,"$AOMP/lib"
fi
run_env ./omp_pgo

# 2. The HSA drain must have produced a device profraw.
shopt -s nullglob
dev=(amdgcn-amd-amdhsa*.profraw)
[ ${#dev[@]} -gt 0 ] || fail "no device (amdgcn-amd-amdhsa) profraw -- HSA drain did not fire"
echo "OmpDevCov: device profraw(s): ${dev[*]}"

# 3. Merge and verify real device counts: classify must have both branches
#    exercised on-device (nonzero, nonzero).
"$PROFDATA" merge -o merged.profdata ./*.profraw
"$PROFDATA" show --all-functions --counts merged.profdata > devcov-profile.txt

# Pull classify's "Block counts: [a, b, ...]" line and require every counter > 0.
bc="$(awk '/[;:]classify:$/ {f=1; next} f && /Block counts:/ {print; exit}' devcov-profile.txt)"
[ -n "$bc" ] || fail "device function classify absent from merged profile (HSA drain did not capture it)"
nums="$(printf '%s\n' "$bc" | grep -oE '[0-9]+')"
[ -n "$nums" ] || fail "could not parse classify block counts from: $bc"
for n in $nums; do
  [ "$n" -gt 0 ] 2>/dev/null || fail "classify has a zero block count ($bc) -- device branch not profiled"
done
echo "OmpDevCov: classify $bc (all branches nonzero)"

# 4. -fprofile-use round trip: must consume the profile without a mismatch /
#    out-of-date diagnostic, then run.
use_log="$(mktemp)"
"$CLANG" -fopenmp --offload-arch="$ARCH" -fprofile-use=merged.profdata \
  omp_pgo.c -o omp_pgo_use -Wl,-rpath,"$AOMP/lib" 2> "$use_log" || {
    echo "--- profile-use build log ---"; cat "$use_log"; rm -f "$use_log"
    fail "-fprofile-use build failed"; }
if grep -Eiq "out of date|profile data may be out of date|no profile data available|mismatch" "$use_log"; then
  echo "--- profile-use build log ---"; cat "$use_log"; rm -f "$use_log"
  fail "-fprofile-use reported a stale/mismatched profile"
fi
rm -f "$use_log"

run_env ./omp_pgo_use

echo "OmpDevCov PASSED (classify branches profiled on device, -fprofile-use round trip clean)"
